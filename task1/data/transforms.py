import torch
import torchvision.transforms.functional as TF
from PIL import Image

def apply_grayscale(img: Image.Image) -> Image.Image:
    """Converts image to grayscale while preserving 3-channel RGB format."""
    gray = TF.to_grayscale(img, num_output_channels=3)
    return gray

def apply_hue_rotation(img: Image.Image, hue_factor: float = 0.5) -> Image.Image:
    """
    Rotates hue by hue_factor in [-0.5, 0.5].
    0.5 corresponds to a 180-degree shift in HSV color space.
    """
    return TF.adjust_hue(img, hue_factor=hue_factor)