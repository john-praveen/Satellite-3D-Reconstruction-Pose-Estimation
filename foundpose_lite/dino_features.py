import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_model_cache = {}


def load_dino(model_name: str, device: str):
    key = (model_name, device)
    if key not in _model_cache:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)
        _model_cache[key] = model
    return _model_cache[key]


def extract_patch_descriptors(image_rgb_uint8: np.ndarray, model_name: str, layer: int,
                               patch_size: int, device: str) -> np.ndarray:
    """image_rgb_uint8: HxWx3 uint8, H and W must be multiples of patch_size.
    Returns (H/patch_size, W/patch_size, C) float32 patch descriptors."""
    model = load_dino(model_name, device)
    h, w = image_rgb_uint8.shape[:2]
    assert h % patch_size == 0 and w % patch_size == 0, \
        f"image size {(h, w)} must be a multiple of patch_size {patch_size}"

    x = torch.from_numpy(image_rgb_uint8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.to(device)

    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=[layer], reshape=True, norm=True)[0]
    # feats: (1, C, H/patch_size, W/patch_size)
    feats = feats[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return feats
