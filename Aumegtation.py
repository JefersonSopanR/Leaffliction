"""
Augmentation.py — Leaffliction Part 2
--------------------------------------
Usage:
    python Augmentation.py <image_path>

Example:
    python Augmentation.py ./images/Apple_healthy/image\ \(1\).JPG

Displays 6 augmented versions of the given image and saves them
in the same directory as the original, named:
    image (1)_Flip.JPG
    image (1)_Rotate.JPG
    image (1)_Skew.JPG
    image (1)_Shear.JPG
    image (1)_Crop.JPG
    image (1)_Distortion.JPG
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# 1. THE 6 AUGMENTATION FUNCTIONS
#    Each takes a PIL Image and returns a PIL Image.
# ─────────────────────────────────────────────

def augment_flip(img: Image.Image) -> Image.Image:
    """
    Mirror the image horizontally (left ↔ right).
    PIL.Image.FLIP_LEFT_RIGHT reflects across the vertical centre axis.
    """
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_rotate(img: Image.Image, angle: float = 25.0) -> Image.Image:
    """
    Rotate the image by `angle` degrees counter-clockwise.

    expand=False keeps the output the same size as the input —
    corners are filled with black (0,0,0).
    expand=True would resize the canvas to fit the rotated image,
    which changes dimensions and breaks batched training.
    """
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))


def augment_skew(img: Image.Image, skew_factor: float = 0.2) -> Image.Image:
    """
    Horizontal skew: shift the top edge right by skew_factor × width.

    Implemented as an affine transform.  An affine transform maps every
    output pixel (x, y) back to a source pixel using a 6-coefficient
    matrix (a, b, c, d, e, f):
        source_x = a*x + b*y + c
        source_y = d*x + e*y + f

    For a horizontal skew, only `b` is non-zero among the off-diagonal
    terms:  source_x = x + b*y   (pixels shift right as y increases).
    """
    w, h   = img.size
    offset = int(skew_factor * w)   # how many pixels the top row shifts

    # Affine coefficients for horizontal skew
    # source_x = 1*x + skew_factor*y + 0
    # source_y = 0*x + 1*y            + 0
    coeffs = (1, skew_factor, 0,
              0, 1,           0)

    return img.transform(
        (w, h),
        Image.AFFINE,
        coeffs,
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def augment_shear(img: Image.Image, shear_factor: float = 0.2) -> Image.Image:
    """
    Vertical shear: shift the right edge down by shear_factor × height.

    Very similar to skew but on the y axis:
        source_x = x
        source_y = shear_factor*x + y

    Skew shifts columns horizontally; shear shifts rows vertically.
    The visual effect is different even though the math is symmetric.
    """
    w, h = img.size

    coeffs = (1,            0, 0,
              shear_factor, 1, 0)

    return img.transform(
        (w, h),
        Image.AFFINE,
        coeffs,
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def augment_crop(img: Image.Image, crop_pct: float = 0.15) -> Image.Image:
    """
    Crop a central region (removing `crop_pct` from each edge) and
    resize it back to the original dimensions.

    This simulates a zoomed-in photo.  The model learns to recognise
    disease from a partial view of the leaf.

    crop_pct=0.15 removes 15% from each side → keeps the central 70%.
    """
    w, h  = img.size
    left  = int(w * crop_pct)
    upper = int(h * crop_pct)
    right = int(w * (1 - crop_pct))
    lower = int(h * (1 - crop_pct))

    cropped = img.crop((left, upper, right, lower))
    return cropped.resize((w, h), Image.LANCZOS)


def augment_distortion(img: Image.Image, strength: float = 20) -> Image.Image:
    """
    Elastic / rubber-sheet distortion.

    How it works:
      1. Generate two random displacement grids (one for x, one for y),
         each the same size as the image.
      2. Smooth those grids with a Gaussian blur so the distortion flows
         naturally instead of looking like random noise.
      3. For every output pixel (x, y), the source pixel is read from
         (x + dx[y,x], y + dy[y,x]).

    `strength` controls the maximum pixel displacement.  12px on a
    256×256 image is visually noticeable but the leaf is still
    recognisable.

    numpy is used here because PIL has no built-in warp operation —
    we need per-pixel coordinate remapping.
    """
    img_np = np.array(img, dtype=np.float32)   # H × W × C  (or H × W for grayscale)
    h, w   = img_np.shape[:2]
    # Random displacement fields
    rng = np.random.default_rng(seed=42)       # fixed seed → reproducible output
    dx  = rng.uniform(-strength, strength, (h, w)).astype(np.float32)
    dy  = rng.uniform(-strength, strength, (h, w)).astype(np.float32)

    print(f"dx[100, 90:100] -> \n{dx[100,90:100]}")

    # Gaussian smoothing — turns random noise into a smooth warp field m
    dx = _gaussian_blur_2d(dx, 3)
    dy = _gaussian_blur_2d(dy, 3)

    print(f"dx[100, 90:100] -> \n{dx[100,90:100]}")
    #print(f"dy[:2, :2] -> \n{dy[:2,:2]}")
    # Build the remapping grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    src_x = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
    src_y = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)

    # Nearest-neighbour remap (fast; bilinear would need more code)
    src_x_idx = src_x.round().astype(int)
    src_y_idx = src_y.round().astype(int)

    if img_np.ndim == 3:
        distorted = img_np[src_y_idx, src_x_idx, :]
    else:
        distorted = img_np[src_y_idx, src_x_idx]

    return Image.fromarray(distorted.astype(np.uint8))


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a simple Gaussian blur to a 2-D float array.
    We implement it as two separable 1-D convolutions (rows then columns),
    which is O(n·k) instead of O(n·k²).

    This avoids pulling in scipy just for one blur call.
    """
    kernel_size = max(3, int(6 * sigma) | 1)   # odd size, at least 3
    half        = kernel_size // 2
    x           = np.arange(-half, half + 1, dtype=np.float32)
    kernel      = np.exp(-0.5 * (x / sigma) ** 2)
    kernel     /= kernel.sum()
    #print("Kernel -> ", kernel)

    # Blur rows
    blurred = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=arr
    )
    # Blur columns
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=blurred
    )
    return blurred


# ─────────────────────────────────────────────
# 2. REGISTRY — maps name → function
#    Order here determines display order.
# ─────────────────────────────────────────────

AUGMENTATIONS = {
    "Flip":       augment_flip,
    "Rotate":     augment_rotate,
    "Skew":       augment_skew,
    "Shear":      augment_shear,
    "Crop":       augment_crop,
    "Distortion": augment_distortion,
}


# ─────────────────────────────────────────────
# 3. SAVE — write augmented images to disk
# ─────────────────────────────────────────────

def save_augmentations(
    original_path: Path,
    augmented: dict[str, Image.Image],
) -> list[Path]:
    """
    Save each augmented image next to the original.
    File name = original_stem + "_" + aug_name + original_suffix
    e.g.  image (1).JPG  →  image (1)_Flip.JPG
    Returns list of saved paths.
    """
    stem      = original_path.stem       # "image (1)"
    suffix    = original_path.suffix     # ".JPG"
    directory = original_path.parent     # "./images/Apple_healthy"

    saved = []
    for name, img in augmented.items():
        out_path = directory / f"{stem}_{name}{suffix}"
        saved.append(out_path)

    return saved


# ─────────────────────────────────────────────
# 4. DISPLAY — show original + 6 augmentations
# ─────────────────────────────────────────────

def display_augmentations(
    original: Image.Image,
    augmented: dict[str, Image.Image],
    image_name: str,
) -> None:
    """
    Show a 2-row, 4-column grid:
      [Original] [Flip] [Rotate] [Skew]
      [Shear]    [Crop] [Distortion]  (last cell empty)
    """
    all_images = {"Original": original} | augmented   # Python 3.9+ dict merge

    n_cols = 2
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 8),
        facecolor="#1A1A2E",
    )
    fig.suptitle(
        f"Data Augmentation — {image_name}",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    items = list(all_images.items())

    for i, ax in enumerate(axes.flat):
        while i < len(items):
            name, img = items[i]
            if name == "Original" or name == "Distortion":
                ax.imshow(img)
                ax.set_title(name, color="white", fontsize=11, pad=6)
                ax.axis("off")
                ax.set_facecolor("#0F0F23")
                break
            i += 1

    plt.tight_layout()
    plt.savefig('jajaja')


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path>")
        print("Example: python Augmentation.py ./images/Apple_healthy/image\\ \\(1\\).JPG")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Error: '{image_path}' is not a valid file.")
        sys.exit(1)

    # Load the original image
    original = Image.open(image_path).convert("RGB")
    print(f"Loaded: {image_path.name}  ({original.size[0]}×{original.size[1]})")

    # Apply all 6 augmentations
    augmented = {name: fn(original) for name, fn in AUGMENTATIONS.items()}

    # Save to disk
    print(f"\nSaving augmented images to: {image_path.parent}/")
    #save_augmentations(image_path, augmented)

    # Display
    display_augmentations(original, augmented, image_path.name)


if __name__ == "__main__":
    main()