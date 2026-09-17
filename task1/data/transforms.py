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

def apply_translation(img: Image.Image, delta: int, direction: str) -> Image.Image:
    """
    Translates an image by delta pixels along a cardinal direction using
    reflection padding followed by a shifted crop to preserve 224x224 size.
    """
    if delta == 0:
        return img

    w, h = img.size  # 224, 224
    # Pad all sides by delta using reflection
    padded = TF.pad(img, padding=[delta, delta, delta, delta], padding_mode="reflect")

    # Shifted crop coordinates: (top, left, height, width)
    if direction == "left":
        # Object shifts left -> camera crops to the right
        crop_box = (delta, 2 * delta, h, w)
    elif direction == "right":
        # Object shifts right -> camera crops to the left
        crop_box = (delta, 0, h, w)
    elif direction == "up":
        # Object shifts up -> camera crops downward
        crop_box = (2 * delta, delta, h, w)
    elif direction == "down":
        # Object shifts down -> camera crops upward
        crop_box = (0, delta, h, w)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return TF.crop(padded, *crop_box)