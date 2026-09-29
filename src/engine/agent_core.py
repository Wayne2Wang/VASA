# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from __future__ import annotations

# pyre-unsafe

import copy
import json
import os
import re
from uuid import uuid4

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image

from ..vlm import EmptyVLMResponse
from .viz import visualize
from ..utils.overlay_style import (
    DEFAULT_MASK_OVERLAY_ALPHA,
    DEFAULT_MASK_OVERLAY_BGR,
    DEFAULT_MASK_OVERLAY_IMAGE_DIM,
    overlay_mask_bgr,
)


def _coerce_mask_index_list(value):
    """Normalize VLM tool args to a list of mask indices.

    Some models (e.g. Llama 4 Scout) emit JSON-encoded strings like \"[1]\" or
    \"[1, 2]\" instead of real arrays. Iterating those strings yields characters
    and silently drops all selections.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _coerce_param_value(raw: str):
    """Parse a tool arg value; keep plain strings if not JSON."""
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _parse_tool_call_from_generated_text(generated_text: str):
    """Parse one tool call into ``{\"name\", \"parameters\"}``.

    Supported dialects:
    - VASA / most VLMs: ``<tool>{\"name\": ..., \"parameters\": {...}}</tool>``
    - Ling-style: ``<tool_call>name\\n<arg_key>k</arg_key>\\n<arg_value>v</arg_value>...``
    - Ling-style JSON args: ``<tool_call>name\\n{...}</tool_call>``
    - Shorthand: ``<tool>name {...}</tool>``
    """
    if not generated_text:
        return None

    # --- Ling <tool_call> ... </tool_call> (tolerate missing/odd close tags) ---
    m = re.search(
        r"<tool_call>\s*(.*?)(?:</tool_call>|</tool>|$)",
        generated_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        body = m.group(1).strip()
        if body.startswith("{"):
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and "name" in obj:
                return {
                    "name": obj["name"],
                    "parameters": obj.get("parameters", {}) or {},
                }

        name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", body)
        if not name_match:
            return None
        name = name_match.group(1)
        rest = body[name_match.end() :].strip()

        keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", rest, flags=re.DOTALL)
        vals = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", rest, flags=re.DOTALL)
        if keys and len(keys) == len(vals):
            return {
                "name": name,
                "parameters": {
                    k.strip(): _coerce_param_value(v) for k, v in zip(keys, vals)
                },
            }

        json_match = re.search(r"\{.*\}", rest, flags=re.DOTALL)
        if json_match:
            try:
                params = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return None
            if isinstance(params, dict):
                if "name" in params and "parameters" in params:
                    return {
                        "name": params["name"],
                        "parameters": params.get("parameters", {}) or {},
                    }
                return {"name": name, "parameters": params}
        return None

    # --- Standard <tool> ... </tool> ---
    if "<tool>" not in generated_text:
        return None

    clipped = generated_text.split("</tool>", 1)[0] + "</tool>"
    inner = (
        clipped.split("<tool>")[-1]
        .split("</tool>")[0]
        .strip()
        .replace(r"}}}", r"}}")  # remove extra } if any
    )
    try:
        obj = json.loads(inner)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "name": obj["name"],
                "parameters": obj.get("parameters", {}) or {},
            }
    except json.JSONDecodeError:
        pass

    # <tool>name {...}</tool>
    name_match = re.match(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})\s*$", inner, flags=re.DOTALL
    )
    if name_match:
        try:
            params = json.loads(name_match.group(2))
        except json.JSONDecodeError:
            return None
        if isinstance(params, dict):
            return {"name": name_match.group(1), "parameters": params}
    return None


def _validate_tool_call(call, candidate_count):
    """Validate before changing state; normalize supported index encodings."""
    fields = {
        'set_strategy': {'text'}, 'segment_phrase': {'text_prompt'},
        'update_working_mask': {'operation', 'selected_masks'},
        'examine_each_mask': set(), 'return_final_output': set(), 'report_no_mask': set(),
    }
    name, params = call.get('name'), call.get('parameters')
    if not isinstance(name, str) or name not in fields:
        return 'Unknown tool name. Use one of the tools in the system instructions.'
    if not isinstance(params, dict) or set(params) != fields[name]:
        return f'{name} requires exactly these parameter fields: {sorted(fields[name])}.'
    for key in ('text', 'text_prompt'):
        if key in params and (not isinstance(params[key], str) or not params[key].strip()):
            return f'{key} must be a nonempty string.'
    if name in {'update_working_mask', 'examine_each_mask'} and candidate_count is None:
        return 'Generate segmentation candidates first.'
    if name == 'update_working_mask':
        op = params['operation']
        if not isinstance(op, str) or op.lower() not in {'add', 'remove', 'replace'}:
            return 'operation must be add, remove, or replace.'
        indices = params['selected_masks']
        if isinstance(indices, str):
            try:
                indices = json.loads(indices)
            except json.JSONDecodeError:
                return 'selected_masks must be an array of integer candidate indices.'
        if type(indices) is int:
            indices = [indices]
        if not isinstance(indices, list) or any(type(i) is not int for i in indices):
            return 'selected_masks must be an array of integer candidate indices.'
        if any(i < 1 or i > candidate_count for i in indices):
            return f'selected_masks must reference current candidates 1 through {candidate_count}.'
        params['selected_masks'] = indices
    return None


def count_images(messages):
    """Count the total number of images present in the messages history."""
    total = 0
    for message in messages:
        # Check if message has content (should be a list)
        if "content" in message and isinstance(message["content"], list):
            # Iterate through each content item
            for content_item in message["content"]:
                # Check if content item is a dict with type "image"
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    total += 1
    return total


def pretty_print_messages(messages, output_path):
    """Pretty print the messages to a file."""
    with open(output_path, "w") as f:
        json.dump(messages, f, indent=4)


def save_working_mask_overlay(
    image_path,
    working_mask_rle,
    output_path,
    *,
    color_bgr: tuple[int, int, int] = DEFAULT_MASK_OVERLAY_BGR,
    overlay_alpha: float = DEFAULT_MASK_OVERLAY_ALPHA,
    image_dim: float = DEFAULT_MASK_OVERLAY_IMAGE_DIM,
):
    """Save working-mask frame (same BGR overlay style as demo_single / visualize_three_experiment_predictions)."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image_bgr.shape[:2]

    if working_mask_rle is None:
        out_bgr = overlay_mask_bgr(
            image_bgr,
            np.zeros((h, w), dtype=np.uint8),
            alpha=0.0,
            color_bgr=color_bgr,
            image_dim=image_dim,
        )
        cv2.imwrite(output_path, out_bgr)
        return

    if isinstance(working_mask_rle, dict):
        rle = working_mask_rle
    else:
        rle = {"size": [h, w], "counts": working_mask_rle}

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]

    out_bgr = overlay_mask_bgr(
        image_bgr,
        mask,
        alpha=overlay_alpha,
        color_bgr=color_bgr,
        image_dim=image_dim,
    )
    cv2.imwrite(output_path, out_bgr)


def _prune_messages_for_next_round(messages_list):
    """
    Return a new messages list that contains only the minimal context needed
    for the next round:
      1) The system prompt.
      2) The raw INPUT IMAGE.
      3) The latest WORKING MASK overlay.
      4) The INITIAL QUERY (as text).
      5) The latest SAM3 OUTPUT (as image).
      6) The latest WARNING message (if any), but only if it appears **after**
         the last assistant tool call (of any tool).
      7) A compact text summary of the **full tool-calling history so far**
         (all assistant tool calls in chronological order).

    All other intermediate chatter is discarded so the model sees a clean,
    structured context each round.
    """
    assert len(messages_list) >= 1

    # Always keep the first system prompt message (index 0).
    keep_indices = {0}

    # Track last indices for the tagged user messages.
    input_image_idx = None
    working_mask_idx = None
    initial_query_idx = None
    sam3_output_idx = None
    strategy_idx = None

    # Track last assistant tool call (any tool) and latest warning after it.
    last_tool_call_idx = None
    warning_after_tool_idx = None

    for idx, msg in enumerate(messages_list):
        role = msg.get("role")
        contents = msg.get("content")
        if not isinstance(contents, list):
            continue

        # User-tagged messages
        if role == "user":
            for content in contents:
                if not (isinstance(content, dict) and content.get("type") == "text"):
                    continue
                text = content.get("text", "")
                if "[INPUT IMAGE]" in text:
                    input_image_idx = idx
                if "[WORKING MASK]" in text:
                    working_mask_idx = idx
                if "[SAM3 OUTPUT]" in text:
                    sam3_output_idx = idx
                if "[INITIAL QUERY]" in text:
                    initial_query_idx = idx
                if "[STRATEGY]" in text:
                    strategy_idx = idx

        # Assistant tool calls (any tool)
        if role == "assistant":
            for content in contents:
                if not (isinstance(content, dict) and content.get("type") == "text"):
                    continue
                text = content.get("text", "")
                if "<tool>" in text:
                    last_tool_call_idx = idx

    # Find the last WARNING message that appears strictly after last_tool_call_idx.
    if last_tool_call_idx is not None:
        for idx in range(len(messages_list) - 1, last_tool_call_idx, -1):
            msg = messages_list[idx]
            if msg.get("role") != "user" or "content" not in msg:
                continue
            for content in msg["content"]:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "text"
                    and isinstance(content.get("text"), str)
                    and "[WARNING]" in content["text"]
                ):
                    warning_after_tool_idx = idx
                    break
            if warning_after_tool_idx is not None:
                break

    # Add indices for the latest tagged messages, if they exist.
    # NOTE: We deliberately do NOT keep the last tool call itself; it is only
    # used as an anchor to determine which WARNING (if any) to keep.
    for idx in (
        input_image_idx,
        working_mask_idx,
        initial_query_idx,
        warning_after_tool_idx,
        sam3_output_idx,
        strategy_idx,
    ):
        if idx is not None:
            keep_indices.add(idx)

    # Build the pruned list in the original appearance order.
    ordered_indices = sorted(keep_indices)
    new_messages = [copy.deepcopy(messages_list[i]) for i in ordered_indices]

    # Reconstruct a compact tool-call history from all assistant messages.
    tool_history = []
    for msg in messages_list:
        if msg.get("role") != "assistant":
            continue
        contents = msg.get("content")
        if not isinstance(contents, list):
            continue
        for content in contents:
            if not (isinstance(content, dict) and content.get("type") == "text"):
                continue
            text = content.get("text", "")
            if "<tool>" not in text or "</tool>" not in text:
                continue
            # Best-effort JSON extraction; ignore failures.
            try:
                snippet = text.split("</tool>", 1)[0] + "</tool>"
                json_str = (
                    snippet.split("<tool>")[-1]
                    .split("</tool>")[0]
                    .strip()
                    .replace(r"}}}", r"}}")
                )
                call = json.loads(json_str)
                name = call.get("name", "UNKNOWN_TOOL")
                params = call.get("parameters", {})
                tool_history.append((name, params))
            except Exception:
                continue

    if tool_history:
        # Format a concise, chronological history (oldest to newest).
        lines = []
        for idx, (name, params) in enumerate(tool_history, start=1):
            # Special-case segment_phrase to expose text_prompt clearly.
            if name == "segment_phrase" and isinstance(params, dict):
                tp = params.get("text_prompt")
                if isinstance(tp, str):
                    param_str = f'text_prompt="{tp}"'
                else:
                    param_str = "text_prompt=<invalid>"
            else:
                # Keep other parameter prints short.
                try:
                    param_str = json.dumps(params)
                except Exception:
                    param_str = "<unserializable>"
            lines.append(f"{idx}. {name}({param_str})")

        history_text = "[INFO] Tool call history (oldest to newest):\n" + "\n".join(
            lines
        )
        new_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": history_text,
                    }
                ],
            }
        )

    # Ensure we never exceed 4 images: raw input + working mask + SAM3 output + initial query.
    assert count_images(new_messages) <= 4
    return new_messages


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    send_generate_request,
    call_sam_service,
    max_generations: int = 20,
    max_tokens: int = 10000,
    output_dir="../../sam3_agent_out",
    print_func=print,
    *,
    working_mask_overlay_bgr: tuple[int, int, int] = DEFAULT_MASK_OVERLAY_BGR,
    working_mask_overlay_alpha: float = DEFAULT_MASK_OVERLAY_ALPHA,
    working_mask_overlay_image_dim: float = DEFAULT_MASK_OVERLAY_IMAGE_DIM,
    system_prompt_path: str | None = None,
    iterative_checking_system_prompt_path: str | None = None,
    enable_error_recovery: bool = True,
    snapshot_callback=None,
):
    """
    Given a text prompt and an image, this tool will perform all aspects of agentic problem solving,
    while saving sam3 and MLLM outputs to their respective directories.

    Args:
        img_path: Path to the input image
        initial_text_prompt: Initial text prompt from the user
        max_generations: Maximum number of send_generate_request calls allowed (default: 20)
    """

    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    print = print_func
    
    # setup dir
    sam_output_dir = os.path.join(output_dir, "sam_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MLLM_SYSTEM_PROMPT_PATH = system_prompt_path or os.path.join(
        current_dir, "system_prompts/system_prompt.txt"
    )
    ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH = iterative_checking_system_prompt_path or os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking.txt"
    )

    # The helper functions are now defined outside the agent_inference function
    with open(MLLM_SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()
    with open(ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH, "r") as f:
        iterative_checking_system_prompt = f.read().strip()

    # init variables
    PATH_TO_LATEST_OUTPUT_JSON = ""
    LATEST_SAM3_TEXT_PROMPT = ""
    USED_TEXT_PROMPTS = (
        set()
    )  # Track all previously used text prompts for segment_phrase

    # Working mask state (persists across segment_phrase calls)
    WORKING_MASK_RLE = None  # COCO RLE "counts" string for the current working mask
    WORKING_MASK_PATH_TEMPLATE = os.path.join(sam_output_dir, "working_mask_overlay_{}.png")
    generation_count = 0  # Counter for number of send_generate_request calls
    WORKING_MASK_BOX = None
    current_working_mask_path = WORKING_MASK_PATH_TEMPLATE.format(generation_count)
    save_working_mask_overlay(
        img_path,
        WORKING_MASK_RLE,
        current_working_mask_path,
        color_bgr=working_mask_overlay_bgr,
        overlay_alpha=working_mask_overlay_alpha,
        image_dim=working_mask_overlay_image_dim,
    )
    WORKING_MASK_PATH = current_working_mask_path

    # Construct the initial message list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"[INPUT IMAGE] The above image is the raw input image",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": current_working_mask_path},
                {
                    "type": "text",
                    "text": f"[WORKING MASK] The above image is the initial working mask rendered on the raw input image. It is initialized as empty (no pixels selected).",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[INITIAL QUERY] The initial user input query is: '{initial_text_prompt}'"},
            ],
        }
    ]

    # print the initial text prompt and image path
    print(f"> Text prompt: {initial_text_prompt}")
    print(f"> Image path: {img_path}")

    def finalize(reason):
        # Prefer the current working mask if it exists, otherwise fall back to the
        # latest SAM3 outputs (if any), otherwise return an empty mask.
        if reason == "no_mask":
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {"original_image_path": img_path, "orig_img_h": height, "orig_img_w": width, "pred_boxes": [], "pred_scores": [], "pred_masks": []}
            rendered_final_output = Image.open(img_path).convert("RGB")
        elif WORKING_MASK_RLE is not None:
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [WORKING_MASK_BOX]
                if WORKING_MASK_BOX is not None
                else [],
                "pred_scores": [1.0] if WORKING_MASK_BOX is not None else [],
                "pred_masks": [WORKING_MASK_RLE],
            }
            rendered_final_output = Image.open(WORKING_MASK_PATH).convert("RGB")
        else:
            # Best-effort: return the latest SAM3 outputs if available.
            if PATH_TO_LATEST_OUTPUT_JSON and os.path.exists(
                PATH_TO_LATEST_OUTPUT_JSON
            ):
                try:
                    latest_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                except Exception:
                    latest_outputs = {}
            else:
                latest_outputs = {}
    
            # Normalize into the final_outputs shape used by downstream code.
            if latest_outputs:
                final_outputs = {
                    "original_image_path": latest_outputs.get(
                        "original_image_path", img_path
                    ),
                    "orig_img_h": latest_outputs.get("orig_img_h"),
                    "orig_img_w": latest_outputs.get("orig_img_w"),
                    "pred_boxes": latest_outputs.get("pred_boxes", []),
                    "pred_scores": latest_outputs.get("pred_scores", []),
                    "pred_masks": latest_outputs.get("pred_masks", []),
                }
                try:
                    rendered_final_output = visualize(
                        final_outputs,
                        mask_alpha=working_mask_overlay_alpha,
                        image_dim=working_mask_overlay_image_dim,
                    )
                except Exception:
                    rendered_final_output = Image.open(img_path)
            else:
                height, width = cv2.imread(img_path).shape[:2]
                final_outputs = {
                    "original_image_path": img_path,
                    "orig_img_h": height,
                    "orig_img_w": width,
                    "pred_boxes": [],
                    "pred_scores": [],
                    "pred_masks": [],
                }
                rendered_final_output = Image.open(img_path)
        
        final_outputs["termination"] = reason
        final_outputs["vlm_calls"] = generation_count
        pruned_messages = _prune_messages_for_next_round(messages)
        return pruned_messages, messages, final_outputs, rendered_final_output

    def request_response(context):
        nonlocal generation_count
        for attempt in range(3):
            try:
                response = send_generate_request(context, max_tokens=max_tokens)
            except EmptyVLMResponse as exc:
                generation_count += 1
                if attempt == 2 or generation_count >= max_generations:
                    raise RuntimeError(f"{exc} Empty-response recovery stopped after {attempt + 1} attempts; "
                                       f"{generation_count}/{max_generations} VLM calls used.") from exc
                warning = f"[WARNING] {exc} Retrying ({attempt + 1}/2)."
                print(warning)
                messages.append({'role': 'user', 'content': [{'type': 'text', 'text': warning}]})
                continue
            generation_count += 1
            return response

    generated_text = "Hello World"
    while True:
        if snapshot_callback:
            snapshot_callback(messages)
        if generation_count >= max_generations:
            return finalize("budget_exhausted")

        print("\n\n")
        print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
        print("\n\n")

        # Send a new MLLM generation request
        pruned_messages = _prune_messages_for_next_round(messages)
        generated_text = request_response(pruned_messages)
        if generated_text is None:
            raise ValueError(
                "MLLM generation request returned None. Check the VLM server, "
                "OpenAI client compatibility, or token settings."
            )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": generated_text}]})
        print(f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n")

        # Parse VASA <tool>{JSON}</tool> or Ling <tool_call>… dialects.
        tool_call = _parse_tool_call_from_generated_text(generated_text)
        if tool_call is None:
            if not enable_error_recovery:
                raise ValueError(
                    f"Generated text does not contain a parsable tool call: {generated_text}"
                )
            _msg = (
                "[WARNING] Your last response did not include a parsable tool call. "
                "Please strictly follow the system instructions: think inside <think>...</think> "
                "and then emit exactly one tool call inside <tool>...</tool>. "
                "The tool call must be in a parsable and structured JSON format. "
            )
            messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")
            continue

        candidate_count = None
        if PATH_TO_LATEST_OUTPUT_JSON:
            with open(PATH_TO_LATEST_OUTPUT_JSON) as f:
                candidate_count = len(json.load(f)['pred_masks'])
        validation_error = _validate_tool_call(tool_call, candidate_count)
        if validation_error:
            if not enable_error_recovery:
                raise ValueError(validation_error)
            messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'[WARNING] {validation_error}'}]})
            continue

        # Call the tool
        if tool_call["name"] == "segment_phrase":
            _msg = "[INFO] Calling segment_phrase tool..."
            # messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")

            # Check if the tool call is valid
            if list(tool_call["parameters"].keys()) != ["text_prompt"]:
                if not enable_error_recovery:
                    raise ValueError(
                        "segment_phrase parameters must contain only 'text_prompt'; "
                        f"got {tool_call['parameters']}"
                    )
                _msg = f"[WARNING] The tool call is invalid. The tool call parameters should be a dictionary with a single key 'text_prompt'. The tool call parameters are: {tool_call['parameters']}"
                messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
                print(f"> {_msg}")
                continue

            # Get the text prompt
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            if current_text_prompt in USED_TEXT_PROMPTS:
                if not enable_error_recovery:
                    raise ValueError(
                        f"Repeated segment_phrase text_prompt: {current_text_prompt}"
                    )
                duplicate_prompt_message = f"[WARNING] You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                messages.append({"role": "user", "content": [{"type": "text", "text": duplicate_prompt_message}]})
                print(f"> {duplicate_prompt_message}")
                continue
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                PATH_TO_LATEST_OUTPUT_JSON = call_sam_service(
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    output_folder_path=sam_output_dir,
                    image_dim=working_mask_overlay_image_dim,
                    mask_alpha=working_mask_overlay_alpha,
                )
                sam3_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                sam3_output_image_path = sam3_outputs["output_image_path"]
                num_masks = len(sam3_outputs["pred_boxes"])

                if num_masks == 0:
                    sam3_output_text_message = f"[SAM3 OUTPUT] No candidates available. [WARNING] The segment_phrase tool did not generate any masks for the text_prompt '{current_text_prompt}'. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append({"role": "user", "content": [{"type": "text", "text": sam3_output_text_message}]})
                    print(f"> {sam3_output_text_message}")
                    continue
                else:
                    sam3_output_text_message = rf"[SAM3 OUTPUT] The segment_phrase tool generated {num_masks} available masks. All {num_masks} available masks are rendered in the image below, now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image, current working mask, and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                    _msg = {"role": "user", "content": [{"type": "text", "text": sam3_output_text_message}, {"type": "image", "image": sam3_output_image_path}]}
                    messages.append(_msg)
                    print(f"> {sam3_output_text_message}")

        elif tool_call["name"] == "set_strategy":
            _msg = "[INFO] Calling set_strategy tool..."
            # messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")

            # Record the agent's high-level strategy decision so it persists in pruned_messages.
            params = tool_call.get("parameters", {})
            # Expect a single plain-text field describing the strategy.
            strategy_text = params.get("text", "")
            if not isinstance(strategy_text, str):
                strategy_text = str(strategy_text)
            strategy_text = strategy_text.strip()
            if not strategy_text:
                strategy_text = "No strategy description was provided."

            _msg_text = f"[STRATEGY] {strategy_text}"
            _msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _msg_text,
                    }
                ],
            }
            messages.append(_msg)
            print(f"> {_msg_text}")

        elif tool_call["name"] == "examine_each_mask":
            _msg = "[INFO] Calling examine_each_mask tool..."
            # messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")

            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            num_masks = len(current_outputs["pred_masks"])
            masks_to_keep = []

            # MLLM check the mask one by one
            for i in range(num_masks):
                if generation_count >= max_generations:
                    # Preserve uninspected candidates instead of silently rejecting them.
                    masks_to_keep.extend(range(i, num_masks))
                    break
                _msg = f"[INFO] Checking mask {i + 1}/{num_masks}..."
                # messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
                print(f"> {_msg}")
                image_w_mask_i, image_w_zoomed_in_mask_i = visualize(
                    current_outputs,
                    i,
                    mask_alpha=working_mask_overlay_alpha,
                    image_dim=working_mask_overlay_image_dim,
                )

                inspection_id = uuid4().hex
                image_w_zoomed_in_mask_i_path = os.path.join(sam_output_dir, f"{inspection_id}_zoom.png")
                image_w_mask_i_path = os.path.join(sam_output_dir, f"{inspection_id}_mask.png")
                image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                image_w_mask_i.save(image_w_mask_i_path)

                iterative_checking_messages = [
                    {"role": "system", "content": iterative_checking_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The raw input image: "},
                            {"type": "image", "image": img_path},
                            {
                                "type": "text",
                                "text": f"The initial user input query is: '{LATEST_SAM3_TEXT_PROMPT}'",
                            },
                            {
                                "type": "text",
                                "text": f"Image with the predicted segmentation mask rendered on it: ",
                            },
                            {"type": "image", "image": image_w_mask_i_path},
                            {
                                "type": "text",
                                "text": f"Image with the zoomed-in mask: ",
                            },
                            {"type": "image", "image": image_w_zoomed_in_mask_i_path},
                        ],
                    },
                ]
                checking_generated_text = request_response(iterative_checking_messages)
                if not checking_generated_text:
                    raise RuntimeError("VLM returned no inspection response")

                messages.extend(iterative_checking_messages)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": checking_generated_text}]})

                print(f"> Generated text for mask {i + 1}: {checking_generated_text}")
                verdict = (
                    checking_generated_text.split("<verdict>")[-1]
                    .split("</verdict>")[0]
                    .strip()
                )
                if "Accept" in verdict and "Reject" not in verdict:
                    _msg = f"[INFO] Mask {i + 1} accepted, keeping it in the outputs."
                    masks_to_keep.append(i)
                elif "Reject" in verdict and "Accept" not in verdict:
                    _msg = f"[INFO] Mask {i + 1} rejected, removing it from the outputs."
                else:
                    if not enable_error_recovery:
                        raise ValueError(
                            f"Unexpected verdict in generated text: {checking_generated_text}. "
                            "Expected 'Accept' or 'Reject'."
                        )
                    _msg = f"[WARNING] Unexpected verdict in generated text: {checking_generated_text}. Expected 'Accept' or 'Reject'. Rejecting this mask."
                    messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
                print(f"> {_msg}")

            updated_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                "pred_scores": [
                    current_outputs["pred_scores"][i] for i in masks_to_keep
                ],
                "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
            }

            image_w_check_masks = visualize(
                updated_outputs,
                mask_alpha=working_mask_overlay_alpha,
                image_dim=working_mask_overlay_image_dim,
            )
            image_w_check_masks_path = os.path.join(sam_output_dir, f"{uuid4().hex}_selected.png")
            image_w_check_masks.save(image_w_check_masks_path)

            if len(masks_to_keep) == 0:
                _msg = {"role": "user", "content": [{"type": "text", "text": f"[SAM3 OUTPUT] No candidates available. [WARNING] The original user query was: '{initial_text_prompt}'. The examine_each_mask tool examined and rejected all of the masks generated by the segment_phrase tool. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt."}]}
                messages.append(_msg)
            else:
                _msg = {"role": "user", "content": [{"type": "text", "text": f"[SAM3 OUTPUT] The original user query was: '{initial_text_prompt}'. After calling the examine_each_mask tool on the available masks, the number of available masks is now {len(masks_to_keep)}. All {len(masks_to_keep)} available masks are rendered in this image below, now you must analyze the {len(masks_to_keep)} available mask(s) carefully, compare them against the raw input image, current working mask, and the original user query, and determine your next action."}, {"type": "image", "image": image_w_check_masks_path}]}
                messages.append(_msg)

            PATH_TO_LATEST_OUTPUT_JSON = os.path.join(sam_output_dir, f"{uuid4().hex}.json")
            json.dump(updated_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)

        elif tool_call["name"] == "update_working_mask":
            _msg = "[INFO] Calling update_working_mask tool..."
            # messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")

            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            h = current_outputs["orig_img_h"]
            w = current_outputs["orig_img_w"]

            params = tool_call.get("parameters", {})
            operation = params.get("operation", "").lower()
            selected_masks = _coerce_mask_index_list(params.get("selected_masks", []))

            if operation not in {"replace", "add", "remove"}:
                if not enable_error_recovery:
                    raise ValueError(
                        "update_working_mask operation must be one of "
                        f"'replace', 'add', or 'remove'; got {operation}"
                    )
                _msg = f"[WARNING] The operation is invalid. The operation should be one of 'replace', 'add', or 'remove'. The operation is: {operation}"
                messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
                print(f"> {_msg}")
                continue

            # Decode selected masks from the current SAM3 outputs into a single binary mask.
            selected_mask_bin = np.zeros((h, w), dtype=bool)
            for idx in selected_masks:
                if not isinstance(idx, int):
                    continue
                if idx < 1 or idx > len(current_outputs["pred_masks"]):
                    continue
                rle = {
                    "size": [h, w],
                    "counts": current_outputs["pred_masks"][idx - 1],
                }
                m = mask_utils.decode(rle)
                if m.ndim == 3:
                    m = m[:, :, 0]
                selected_mask_bin |= (m > 0)

            # Decode existing working mask if present.
            if WORKING_MASK_RLE is not None:
                wm = mask_utils.decode(
                    {"size": [h, w], "counts": WORKING_MASK_RLE}
                )
                if wm.ndim == 3:
                    wm = wm[:, :, 0]
                working_bin = wm > 0
            else:
                working_bin = np.zeros((h, w), dtype=bool)

            # Apply the requested operation.
            if operation == "replace":
                new_bin = selected_mask_bin
            elif operation == "add":
                new_bin = working_bin | selected_mask_bin
            elif operation == "remove":
                new_bin = working_bin & (~selected_mask_bin)
            else:
                # Should not happen due to assert above.
                new_bin = working_bin

            # Re-encode the updated working mask to COCO RLE.
            new_uint8 = new_bin.astype(np.uint8)
            rle_encoded = mask_utils.encode(np.asfortranarray(new_uint8))
            counts = rle_encoded["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("ascii")
            WORKING_MASK_RLE = counts

            # Compute bounding box [x1, y1, x2, y2] for the working mask, if any pixels are set.
            ys, xs = np.where(new_bin)
            if ys.size == 0 or xs.size == 0:
                # The updated working mask is empty; no bounding box.
                WORKING_MASK_BOX = None
            else:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                WORKING_MASK_BOX = [x1, y1, x2, y2]

            current_working_mask_path = WORKING_MASK_PATH_TEMPLATE.format(generation_count)
            save_working_mask_overlay(
                img_path,
                WORKING_MASK_RLE,
                current_working_mask_path,
                color_bgr=working_mask_overlay_bgr,
                overlay_alpha=working_mask_overlay_alpha,
                image_dim=working_mask_overlay_image_dim,
            )
            WORKING_MASK_PATH = current_working_mask_path
            messages.append({"role": "user", "content": [
                {"type": "image", "image": current_working_mask_path},
                {"type": "text", "text": f"[WORKING MASK] Generation {generation_count}: The current working mask is rendered in the image above: "},
            ]})


        elif tool_call["name"] == "return_final_output":
            return finalize("completed")

        elif tool_call["name"] == "report_no_mask":
            return finalize("no_mask")

        else:
            if not enable_error_recovery:
                raise ValueError(f"Unknown tool call: {tool_call['name']}")
            _msg = f"[WARNING] Unknown tool call: {tool_call['name']}. Please strictly follow the system instructions: think inside <think>...</think> and then emit exactly one tool call inside <tool>...</tool>."
            messages.append({"role": "user", "content": [{"type": "text", "text": _msg}]})
            print(f"> {_msg}")
            continue


