import cv2
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamAutomaticMaskGenerator
)
import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import tqdm
import torchvision.transforms as T

sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Draw and save functions
def save_mask_data(output_dir, mask_list, filename="mask", grayscale=False):
    ext = ".png"
    mask_img = torch.zeros(mask_list[0]['segmentation'].shape[-2:], dtype=torch.uint8)
    
    for idx, mask in enumerate(mask_list):
        mask_img[mask['segmentation'] > 0] = idx + 1
        
    mask_img = mask_img.numpy() * 255

    if not grayscale:
        uint_to_rgb_dict = {}

        # Get unique uint values
        unique_uints = np.unique(mask_img)

        # For each unique uint, generate a random RGB color
        for uint in unique_uints:
            uint_to_rgb_dict[uint] = np.random.randint(0, 256, 3)

        # Create an empty RGB image
        rgb_image = np.zeros((*mask_img.shape, 3), dtype=np.uint8)

        # Map each uint in the input array to its corresponding RGB color
        for i in range(mask_img.shape[0]):
            for j in range(mask_img.shape[1]):
                rgb_image[i, j] = uint_to_rgb_dict[mask_img[i, j]]

        mask_img = rgb_image

    # kernel = np.ones((5,5), np.uint8)
    # mask_img = cv2.morphologyEx(mask_img.numpy(), cv2.MORPH_CLOSE, kernel, iterations = 3)

    cv2.imwrite(os.path.join(output_dir, filename+ext),
                (mask_img).astype(np.uint8))
    print(f"Saved {os.path.join(output_dir, filename+ext)}")


# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Automatic Segment-Anything Demo", add_help=True)

    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str,
                        required=True, help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--grayscale", action="store_true", help="Save masks as grayscale incremental values")
    parser.add_argument("--overview", action="store_true",
                        help="Display boxes and labels over input images")

    parser.add_argument("--device", type=str, default="cuda",
                        help="running on cpu only!, default=cpu")
    args = parser.parse_args()

    # cfg
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    output_dir = args.output_dir
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    if args.overview:
        os.makedirs(os.path.join(output_dir, "comp"), exist_ok=True)

    # initialize SAM
    if use_sam_hq:
        auto_predictor = SamAutomaticMaskGenerator(
            sam_hq_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    else:
        auto_predictor = SamAutomaticMaskGenerator(
            sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # ignore subfolder and raw image
    # EXT_IGNORE = ['.dng','.json','.raw','.cr2','.cr3'] # maybe more
    EXT_IMG_ALLOW = ['.jpg', '.jpeg', '.png', '.bmp',
                     '.tiff', '.tif', '.gif', '.heic', '.heif']

    paths = [image_path] if os.path.isfile(image_path) else [os.path.join(image_path, f) for f in os.listdir(
        image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(tuple(EXT_IMG_ALLOW))]
    print(f"Found {len(paths)} images")

    pbar = tqdm.tqdm(total=len(paths))
    masks = {}
    for image_path in sorted(paths):
        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(image_path).split('/')[-1]
        filename_ext = filename.split('.')

        # SETUP IMAGE FOR SAM
        # image = np.array(image)
        H, W, C = image.shape

        # Predict over whole image

        res = auto_predictor.generate(image)
        """
        masks = [{
            segmentation : the mask
            area : the area of the mask in pixels
            bbox : the boundary box of the mask in XYWH format
            predicted_iou : the model's own prediction for the quality of the mask
            point_coords : the sampled input point that generated this mask
            stability_score : an additional measure of mask quality
            crop_box : the crop of the image used to generate this mask in XYWH format
            },
            { ... },
        ]
        """
        if res is None:
            print(f"Failed to segment {filename}")
            continue
        save_mask_data(output_dir, res, filename_ext[0], grayscale=args.grayscale)

        pbar.update(1)
    pbar.close()
    print("Done")  # end of the script
