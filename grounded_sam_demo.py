import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import tqdm

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np


# ignore subfolder and raw image
# EXT_IGNORE = ['.dng','.json','.raw','.cr2','.cr3'] # maybe more
EXT_IMG_ALLOW = ['.jpg','.jpeg','.png','.bmp','.tiff','.tif','.gif', '.heic', '.heif']

## Helper functions
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_norm, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image_norm

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

## Grounding functions
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

## Draw and save functions
def show_mask(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255])
    mask = mask.astype(np.uint8)
    mask = np.transpose(mask, (1, 2, 0))

    mask_index = mask[:, :, 0] > 0
    # simultate transparency
    image[mask_index] = image[mask_index] * 0.5 + color * 255 * 0.5
    
    return image 

def show_box(box, image, label):
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])
    color = (0, 255, 0)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 5)
    cv2.putText(image, label, (x0+5, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    return image

def save_mask_data(output_dir, mask_list, box_list, label_list, filename="mask", binary=False, invert=False):
    ext = ".png"
    
    # Determine the value to use for filling the mask image based on binary and invert settings.
    bin_value = 255 if not binary or invert else 0

    mask_img = torch.ones(mask_list.shape[-2:], dtype=torch.uint8) * (bin_value if not binary else 0)

    for idx, mask in enumerate(mask_list):
        # Use list comprehension to convert the mask tensor to a NumPy array once per iteration.
        np_mask = mask.cpu().numpy()[0]
        if invert:
            np_mask = ~np_mask  # Invert the mask inside the loop for better memory usage.
        mask_img[np_mask] = bin_value * ((idx + 1) / len(mask_list)) if not binary else bin_value

    mask_img = mask_img.numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, filename + ext), mask_img)

    if not binary:
        value = 0
        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        # with open(os.path.join(output_dir, f"{filename.split('.')[0]}.json"), 'w') as f:
        #     json.dump(json_data, f)

# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--config", type=str, default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="checkpoints/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")

    parser.add_argument("--binary_mask", action="store_true", help="save masks as binary image")
    parser.add_argument("--overview", action="store_true", help="save boxes and masks over images")
    parser.add_argument("--invert", action="store_true", help="invert mask")

    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    if args.overview:
        os.makedirs(os.path.join(output_dir, "comp"), exist_ok=True)

    # LOAD MODELS
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    paths = [image_path] if os.path.isfile(image_path) else [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(tuple(EXT_IMG_ALLOW))]
    print(f"Found {len(paths)} images")

    pbar = tqdm.tqdm(total=len(paths))
    for image_path in sorted(paths):
        # 1. load image
        image_pil, image_norm = load_image(image_path)
        filename = os.path.basename(image_path).split('/')[-1]
        filename_ext = filename.split('.')
        
        # 2. Run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image_norm, text_prompt, box_threshold, text_threshold, device=device
        )
        if len(boxes_filt) == 0:
            pbar.set_description(f"[{filename}] - No object found")
            empty_masks = torch.zeros(1, 1, H, W)
            # save_mask_data(output_dir, empty_masks, boxes_filt, pred_phrases, filename_ext[0], binary=args.binary_mask)
            continue
        else:
            pbar.set_description(f"[{filename}] - {len(boxes_filt)} objects found")

        # SETUP IMAGE FOR SAM
        image = np.array(image_pil)
        H, W, C = image.shape
        predictor.set_image(image)

        # SETUP BOXES FOR SAM
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        
        # 3. RUN SAM
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        predictor.reset_image()

        # SAVE MASKs
        if args.overview:
            # draw boxes and masks over images
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for label, box in zip(pred_phrases, boxes_filt):
                image = show_box(box.numpy(), image, label)
            for mask in masks:
                image = show_mask(mask.cpu().numpy(), image, random_color=True)
            cv2.imwrite(os.path.join(output_dir,"comp",f"{filename_ext[0]}_all_masks.jpg"), image)

        save_mask_data(output_dir, masks, boxes_filt, pred_phrases, filename_ext[0], binary=args.binary_mask, invert=args.invert)

        pbar.update(1)
    pbar.close()
    print("Done") # end of the script