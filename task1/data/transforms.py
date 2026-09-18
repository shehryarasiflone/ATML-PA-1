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

def apply_patch_shuffle(img: Image.Image, seed: int) -> Image.Image:
    """
    Divides a 224x224 image into a 4x4 grid (16 patches of 56x56)
    and permutes them using a deterministic non-identity shuffle.
    """
    w, h = img.size
    grid_size = 4
    patch_w = w // grid_size  # 56
    patch_h = h // grid_size  # 56

    patches = []
    for r in range(grid_size):
        for c in range(grid_size):
            box = (c * patch_w, r * patch_h, (c + 1) * patch_w, (r + 1) * patch_h)
            patches.append(img.crop(box))

    rng = np.random.default_rng(seed)
    # Ensure a non-identity permutation
    perm = rng.permutation(16)
    while np.array_equal(perm, np.arange(16)):
        perm = rng.permutation(16)

    shuffled_img = Image.new("RGB", (w, h))
    for idx, p_idx in enumerate(perm):
        r = idx // grid_size
        c = idx % grid_size
        shuffled_img.paste(patches[p_idx], (c * patch_w, r * patch_h))

    return shuffled_img