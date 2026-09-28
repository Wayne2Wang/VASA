"""SAM3 image adapter; no private SAM3 fork or server required."""
import json
from pathlib import Path
from uuid import uuid4


class SAM3Segmenter:
    def __init__(self, checkpoint=None, device='cuda'):
        import torch
        if device == 'cpu':
            from .sam3_cpu import build_cpu_model
            build_model = build_cpu_model
        else:
            from sam3.model_builder import build_sam3_image_model
            build_model = build_sam3_image_model
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for the documented SAM3 setup.')
        kwargs = {} if device == 'cpu' else {'device': device}
        if checkpoint:
            kwargs.update(checkpoint_path=str(checkpoint), load_from_HF=False)
        self.model = build_model(**kwargs).to(device).eval()
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5, device=device)
        self.device = device

    def __call__(self, image_path, text_prompt, output_folder_path, image_dim=0.48, mask_alpha=0.58):
        import numpy as np
        import torch
        from PIL import Image
        from pycocotools import mask as masks
        from .engine.helpers.mask_overlap_removal import remove_overlapping_masks
        from .engine.viz import visualize
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            width, height = image.size
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        encoded = []
        for mask in state['masks'].detach().cpu().numpy():
            mask = mask.reshape(height, width)
            rle = masks.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
            encoded.append(rle['counts'].decode('ascii'))
        boxes = state['boxes'].detach().cpu().numpy()
        normalized = [[float(x1/width), float(y1/height), float((x2-x1)/width), float((y2-y1)/height)] for x1,y1,x2,y2 in boxes]
        result = {'original_image_path': image_path, 'orig_img_h': height, 'orig_img_w': width, 'pred_boxes': normalized, 'pred_masks': encoded, 'pred_scores': state['scores'].detach().cpu().tolist()}
        result = remove_overlapping_masks(result)
        order = sorted(range(len(result['pred_scores'])), key=lambda i: result['pred_scores'][i], reverse=True)
        for key in ['pred_boxes','pred_masks','pred_scores']:
            result[key] = [result[key][i] for i in order]
        # Never use VLM-generated text as a filesystem path.
        folder = Path(output_folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / uuid4().hex
        result['output_image_path'] = str(stem.with_suffix('.png'))
        visualize(result, image_dim=image_dim, mask_alpha=mask_alpha).save(result['output_image_path'])
        path = stem.with_suffix('.json')
        path.write_text(json.dumps(result, indent=2))
        return str(path)
