import sys
from PIL import Image
import argparse
from pathlib import Path
import numpy as np

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tiff", ".JPG", ".JPEG", ".PNG",
}

def Distortion_image(image: Image.Image):

    img_array = np.array(image)

    widht,height = image.size
    grid_w, grid_h = np.meshgrid(range(widht), range(height))

    cut = 125

    mask = (grid_w < cut)
    mask_reverse = (grid_w > cut)
    grid_w[mask] += 255 - cut
    grid_w[mask_reverse] -=  cut
    img_array = img_array[grid_h,grid_w]

    image_array = Image.fromarray(img_array)

    return image_array

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=("Aumegtacion image_file"))
    argparser.add_argument("image_file", type=str, help="image from the image directory")
	
    args = argparser.parse_args()

    image_file = Path(args.image_file)
    if not image_file.is_file() or image_file.suffix not in IMAGE_EXTENSIONS:
        raise Exception(f"{image_file.name} is not a valid file!")

    image = Image.open(image_file).convert('RGB')
    print(image.size)

    image_to_print = Distortion_image(image)
    print(image_to_print.size)
    
    image_to_print.save('image_mirror.png')


