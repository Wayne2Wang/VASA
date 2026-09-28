import base64
import io
import os

from PIL import Image


def get_image_base64_and_mime(image_path, downscale_factor: int = 1):
    downscale_factor = int(downscale_factor)
    if downscale_factor < 1:
        raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor}")

    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")

    if downscale_factor == 1:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8"), mime_type

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        new_w = max(1, w // downscale_factor)
        new_h = max(1, h // downscale_factor)
        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def _encode_message_images(
    messages,
    *,
    downscale_factor: int = 1,
    print_func=print,
    include_detail: bool = True,
):
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for content in message["content"]:
                if isinstance(content, dict) and content.get("type") == "image":
                    image_path = content["image"]
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            image_path, downscale_factor=downscale_factor
                        )
                    except Exception as exc:
                        raise ValueError(f"Could not encode image {image_path}") from exc
                    image_url = {"url": f"data:{mime_type};base64,{base64_image}"}
                    if include_detail:
                        image_url["detail"] = "high"
                    processed_content.append(
                        {"type": "image_url", "image_url": image_url}
                    )
                else:
                    processed_content.append(content)
            processed_message["content"] = processed_content
        processed_messages.append(processed_message)
    return processed_messages


