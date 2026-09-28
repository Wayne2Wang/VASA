"""Single-image VASA inference."""
import json
import os
from datetime import datetime
from .runs import automatic_output
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext


def _portable_trace(value, root):
    """Store image references relative to the run folder; leave prose untouched."""
    if isinstance(value, list):
        return [_portable_trace(item, root) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            is_image = key == 'image' and value.get('type') == 'image'
            if (is_image or key in {'original_image_path', 'output_image_path'}) and isinstance(item, str):
                result[key] = Path(item).resolve().relative_to(root).as_posix()
            else:
                result[key] = _portable_trace(item, root)
        return result
    return value


def segment(image, query, output=None, *, model=None, base_url=None, api_key=None,
            checkpoint=None, device='cuda', max_rounds=20, max_tokens=10000,
            image_downscale=2, reasoning_effort=None, verbose=False, save_trace=True, progress_callback=None, preview_callback=None):
    if not query.strip():
        raise ValueError('Query must not be empty.')
    if min(max_rounds, max_tokens, image_downscale) < 1:
        raise ValueError('Round limit, token limit, and image downscale must be positive.')
    model = model or os.getenv('VASA_MODEL')
    base_url = base_url or os.getenv('VASA_BASE_URL')
    api_key = api_key or os.getenv('VASA_API_KEY')
    if not all((model, base_url, api_key)):
        raise ValueError('Set VASA_MODEL, VASA_BASE_URL, and VASA_API_KEY (or pass them to segment).')
    from PIL import Image, ImageOps
    import numpy as np
    from pycocotools import mask as masks
    from .engine.agent_core import agent_inference
    from .engine.agent_core import save_working_mask_overlay
    from .segmenter import SAM3Segmenter
    from .vlm import VLMClient
    source = Path(image).resolve()
    with Image.open(source) as im:
        normalized = ImageOps.exif_transpose(im).convert('RGB')
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    folder = Path(output).resolve() if output is not None else automatic_output(source if save_trace else "image", query if save_trace else "run").resolve()
    folder.mkdir(parents=True, exist_ok=output is not None)
    if any(folder.iterdir()):
        raise ValueError('Output directory must be empty. Choose a new directory for each run.')
    # A per-output ignore file protects custom output locations inside Git repos.
    (folder / '.gitignore').write_text('*\n')
    workspace = nullcontext(str(folder)) if save_trace else TemporaryDirectory(prefix='vasa-')
    with workspace as workspace_path:
        work = Path(workspace_path).resolve()
        normalized_path = work / 'input.png'
        normalized.save(normalized_path)
        client = VLMClient(base_url=base_url, model=model, api_key=api_key, downscale=image_downscale, reasoning_effort=reasoning_effort)
        sam = SAM3Segmenter(checkpoint=checkpoint, device=device)
        def report_progress(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
            if progress_callback:
                progress_callback(" ".join(str(arg) for arg in args))
        def preview(history):
            if preview_callback:
                from .trace_html import render_markup
                preview_callback(render_markup(work, {'query': query, 'model': model}, _portable_trace(history, work), live=True))
        _, history, result, _ = agent_inference(
            str(normalized_path), query, client, sam, max_generations=max_rounds,
            max_tokens=max_tokens, output_dir=str(work/'trace'),
            print_func=report_progress, snapshot_callback=preview if preview_callback else None)
        h, w = result['orig_img_h'], result['orig_img_w']
        merged = np.zeros((h,w), dtype=np.uint8)
        for counts in result['pred_masks']:
            merged |= masks.decode({'size':[h,w], 'counts':counts}).reshape(h,w).astype(np.uint8)
        Image.fromarray(merged*255).save(folder/'mask.png')
        rle = masks.encode(np.asfortranarray(merged))['counts'].decode('ascii')
        save_working_mask_overlay(str(normalized_path), rle, str(folder/'overlay.png'))
        if save_trace:
            (work/'trace/history.json').write_text(json.dumps(_portable_trace(history, work), indent=2))
            # Runtime consumers use absolute paths; convert candidate records only after inference.
            for record in (work/'trace/sam_out').glob('*.json'):
                data = json.loads(record.read_text())
                record.write_text(json.dumps(_portable_trace(data, work), indent=2))
    metadata = {'created_at':created_at, 'model':model, 'max_rounds':max_rounds,
                'max_tokens':max_tokens, 'image_downscale':image_downscale,
                'reasoning_effort':getattr(client, 'reasoning_effort', reasoning_effort), 'vlm_calls':client.calls,
                'termination':result['termination'], 'usage':client.usage,
                'mask':'mask.png', 'overlay':'overlay.png'}
    if save_trace:
        metadata['query'] = query
        metadata['image_name'] = source.name
        metadata['project_path'] = str(Path(__file__).resolve().parents[1])
        metadata['trace'] = {'history': 'trace/history.json', 'input': 'input.png',
                             'path_base': 'output_directory', 'format_version': 1}
    (folder/'result.json').write_text(json.dumps(metadata, indent=2))
    if save_trace:
        from .trace_html import render_trace
        render_trace(folder)
    return {**metadata, 'output_dir': str(folder)}
