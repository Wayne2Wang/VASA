"""CPU compatibility for the pinned SAM3 image builder, without editing SAM3."""
import importlib.util
import sys
import types
from unittest.mock import patch


def _distance_transform(data):
    import numpy as np
    import torch
    import cv2

    if data.ndim != 3 or data.device.type != 'cpu':
        raise ValueError('CPU distance transform expects a CPU tensor shaped (B, H, W).')
    result = np.empty(tuple(data.shape), dtype=np.float32)
    for i, mask in enumerate(data.detach().numpy()):
        # Match SAM3's finite infinity when the image has no background pixels.
        result[i] = 1e9 if mask.all() else cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return torch.from_numpy(result)


def load_cpu_builder():
    # SAM3 imports this tracking-only CUDA utility even for image inference.
    # Supply a real CPU implementation only when Triton is unavailable.
    if importlib.util.find_spec('triton') is None and 'sam3.model.edt' not in sys.modules:
        module = types.ModuleType('sam3.model.edt')
        module.edt_triton = _distance_transform
        sys.modules[module.__name__] = module
    from sam3 import model_builder
    return model_builder


def build_cpu_model(**kwargs):
    builder = load_cpu_builder()
    original = builder._create_position_encoding

    def position_encoding(precompute_resolution=None):
        # The upstream cache prewarmer allocates CUDA tensors. Eager CPU
        # inference computes the same encodings on demand from input tensors.
        return original(precompute_resolution=None)

    get_coords = builder.TransformerDecoder._get_coords

    def cpu_coords(height, width, device):
        return get_coords(height, width, device='cpu')

    with patch.object(builder, '_create_position_encoding', position_encoding), \
            patch.object(builder.TransformerDecoder, '_get_coords', staticmethod(cpu_coords)):
        model = builder.build_sam3_image_model(device='cpu', **kwargs)
    from sam3.model.vitdet import Mlp
    from sam3.model.geometry_encoders import SequenceGeometryEncoder
    for layer in model.modules():
        if isinstance(layer, Mlp):
            layer.forward = types.MethodType(_cpu_mlp_forward, layer)
        if isinstance(layer, SequenceGeometryEncoder):
            layer._encode_boxes = types.MethodType(_cpu_encode_boxes, layer)
    return model


def _cpu_mlp_forward(self, x):
    # Upstream's fused linear/activation forces bfloat16, while CPU weights
    # remain float32. Use the equivalent ordinary PyTorch operations.
    x = self.act(self.fc1(x))
    x = self.drop1(x)
    x = self.norm(x)
    return self.drop2(self.fc2(x))


def _cpu_encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
    # VASA supplies text prompts only. Upstream nevertheless pins a tensor
    # for zero boxes; pin_memory can crash on an Apple CPU-only torch build.
    if boxes.shape[0] != 0:
        raise ValueError('VASA CPU compatibility supports text prompts, not box prompts.')
    return img_feats.new_empty((0, boxes.shape[1], self.d_model)), boxes_mask
