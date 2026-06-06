import sys
from PIL import Image
import argparse
from pathlib import Path
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tiff", ".JPG", ".JPEG", ".PNG",
}

def flip_image(image: Image.Image)-> Image.Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image(image: Image.Image) -> Image.Image:
    angle = 45
    return image.rotate(angle,resample=Image.BILINEAR)

def skew_image(image: Image.Image):
    w,h = image.size
    data = (1, 0.2, 0,
            0, 1, 0)
    return image.transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR)

def shear_image(image: Image.Image):
    w,h = image.size
    data = (1, 0, 0,
            0.2, 1, 0)
    return image.transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR)

def crop_image(image: Image.Image):
    cut = 0.15

    w, h = image.size
    l = w  * cut
    u = h * cut
    r = w * (1 - cut)
    b = h * (1 - cut)

    crop = image.crop((l, u, r, b))

    return crop.resize((w, h), resample=Image.LANCZOS)

def Distortion_image(image: Image.Image):

    img_array = np.array(image)

    width,height = image.size
    grid_w, grid_h = np.meshgrid(range(width), range(height))

    randonnes = np.random.default_rng(seed=23)

    strengh = 6

    d_w = randonnes.uniform(-strengh , strengh , (width, height))
    d_h = randonnes.uniform(-strengh , strengh , (width, height))

    d_w = gaussian_blur(d_w, sigma=3)
    d_h = gaussian_blur(d_h, sigma=3)

    grid_w = np.clip(grid_w + d_w, 0, width - 1).astype(np.uint8)
    grid_h = np.clip(grid_h + d_h, 0, height - 1).astype(np.uint8)

    img_array = img_array[grid_h, grid_w]


    image_array = Image.fromarray(img_array)

    return image_array

def gaussian_blur(array, sigma):
    kernel_size = np.maximum(6, (sigma * 6) | 1)

    half = kernel_size // 2

    offset = np.arange(-half, half + 1)

    kernel      = np.exp(-0.5 * (offset / sigma) ** 2)

    kernel = kernel/kernel.sum()

    blur = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="same"), axis=1,arr=array)
    blur = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="same"), axis=0, arr=blur)

    return blur


def main():
    argparser = argparse.ArgumentParser(description=("Aumegtacion image_file"))
    argparser.add_argument("image_file", type=str, help="image from the image directory")
	
    args = argparser.parse_args()

    try:

        image_file = Path(args.image_file)
        if not image_file.is_file() or image_file.suffix not in IMAGE_EXTENSIONS:
            raise Exception(f"{image_file.name} is not a valid file!")

        image = Image.open(image_file).convert('RGB')
        print(image.size)

        flip_image(image)

        rotate_image(image)

        skew_image(image)

        shear_image(image)

        crop_image(image)
        
        image_to_print = Distortion_image(image)
        print(image_to_print.size)
        
        image_to_print.save('image.png')
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    


