from __future__ import annotations

import os
import random
from scipy import ndimage

import gradio as gr
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry, sam_hq_model_registry
import numpy as np
import torch


from pathlib import Path
import numpy as np
from PIL import Image as _Image  # using _ to minimize namespace pollution

# Monkey patching the Gallery class to allow the gallery 
# to postprocess an existing list of images in gallery
class Gallery(gr.Gallery):
    # Overriding the postprocess method to return the gallery
    def postprocess(
        self,
        y: list[np.ndarray | _Image.Image | str]
        | list[tuple[np.ndarray | _Image.Image | str, str]]
        | None,
    ) -> list[str]:
        """
        Parameters:
            y: list of images, or list of (image, caption) tuples
        Returns:
            list of string file paths to images in temp directory
        """
        if y is None:
            return []
        output = []
        for img in y:
            caption = None
            if isinstance(img, (tuple, list)):
                img, caption = img
            if isinstance(img, np.ndarray):
                file = self.img_array_to_temp_file(img, dir=self.DEFAULT_TEMP_DIR)
                file_path = str(gr.utils.abspath(file))
                self.temp_files.add(file_path)
            elif isinstance(img, _Image.Image):
                file = self.pil_to_temp_file(img, dir=self.DEFAULT_TEMP_DIR)
                file_path = str(gr.utils.abspath(file))
                self.temp_files.add(file_path)
            elif isinstance(img, (str, Path)):
                if gr.utils.validate_url(img):
                    file_path = img
                else:
                    file_path = self.make_temp_copy_if_needed(img)
            elif isinstance(img, dict):
                if img["is_file"]:
                    file_path = self.make_temp_copy_if_needed(img["name"]) 
                else:
                    file = self.img_array_to_temp_file(img["data"], dir=self.DEFAULT_TEMP_DIR)
                    file_path = str(gr.utils.abspath(file))
                    self.temp_files.add(file_path)
            else:
                raise ValueError(f"Cannot process type as image: {type(img)}")

            if caption is not None:
                output.append(
                    [{"name": file_path, "data": None, "is_file": True}, caption]
                )
            else:
                output.append({"name": file_path, "data": None, "is_file": True})

        return output

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res

def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

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
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=20)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), font_size=40, fill="white")



config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
dino_ckpt = "checkpoints/groundingdino_swint_ogc.pth"
sam_ckpt = "checkpoints/sam_vit_h_4b8939.pth"
output_dir="outputs"
device="cuda"
img_idx = 0
groundingdino_model = None
sam_predictor = None
sam_automask_generator = None

def load_ckpt(dino_path, sam_path, use_sam_hq=False, sam_version="vit_h"):
    global groundingdino_model, sam_predictor, sam_automask_generator

    # load grounding dino model
    try:
        groundingdino_model = load_model(config_file, dino_path, device=device)
    except Exception as e:
        print(f"Error loading Grounding DINO model: {e}")
        return gr.Textbox(label="Error loading model !"), sam_path, use_sam_hq, sam_version, gr.Button(label="Run", interactive=False)

    # load sam model
    try:
        # initialize SAM
        if use_sam_hq:
            sam = sam_hq_model_registry[sam_version](checkpoint=sam_path)
        else:
            sam = sam_model_registry[sam_version](checkpoint=sam_path)
        # sam = build_sam(checkpoint=sam_path)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return dino_path, sam_path.update(label="Error loading model !"), use_sam_hq, sam_version, gr.Button(label="Run", interactive=False)

    return dino_path, sam_path, use_sam_hq, sam_version, gr.Button(label="Run", interactive=True)

def run_grounded_sam(input_image, text_prompt, task_type, box_threshold, text_threshold, gallery, scribble_mode):

    global groundingdino_model, sam_predictor, sam_automask_generator

    if gallery is None:
        gallery = []

    # load image
    if task_type == 'scribble':
        image = input_image["image"]
        scribble = input_image["mask"]
    else:
        image = input_image
    size = image.size # w, h


    filename, ext = paths[img_idx].split('/')[-1].split('.')
    image_pil = image.convert("RGB")
    image = np.array(image_pil)

    if task_type == 'scribble':
        sam_predictor.set_image(image)
        scribble = scribble.convert("RGB")
        scribble = np.array(scribble)
        scribble = scribble.transpose(2, 1, 0)[0]

        # 将连通域进行标记
        labeled_array, num_features = ndimage.label(scribble >= 255)

        # 计算每个连通域的质心
        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
        centers = np.array(centers)

        point_coords = torch.from_numpy(centers)
        point_coords = sam_predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
        point_coords = point_coords.unsqueeze(0).to(device)
        point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)
        if scribble_mode == 'split':
            point_coords = point_coords.permute(1, 0, 2)
            point_labels = point_labels.permute(1, 0)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            mask_input = None,
            boxes = None,
            multimask_output = False,
        )
    elif task_type == 'automask':
        masks = sam_automask_generator.generate(image)
    else:
        transformed_image = transform_image(image_pil)

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
        )

        # process boxes
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()


        if task_type == 'seg':
            sam_predictor.set_image(image)

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

    if task_type == 'det':
        image_draw = ImageDraw.Draw(image_pil)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)
        return gallery + [(image_pil, f'{filename}_bbox.{ext}')]
    
    elif task_type == 'automask':
        full_img, res = show_anns(masks)
        return gallery + [(full_img, f'{filename}_automask.{ext}')]
    
    elif task_type == 'scribble':
        mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return gallery + [(image_pil,filename), (mask_image,f'{filename}_mask.{ext}')]
    
    elif task_type == 'seg':
        mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
        image_draw = ImageDraw.Draw(image_pil)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return gallery + [(image_pil, f'{filename}_mask_bbox.{ext}'), (mask_image, f'{filename}_mask.{ext}')]
    
    else:
        print("task_type:{} error!".format(task_type))

def next_img():
    global img_idx

    if img_idx < len(paths) - 2:
        img_idx += 1
        next_btn = gr.Button(value="Next Image")  
    elif img_idx == len(paths) - 2:
        img_idx += 1
        next_btn = gr.Button(value="No more Image", interactive=False)
    else: #last image ; func should be disabled
        next_btn =  gr.Button(value="No more images, IMP", interactive=False)

    return Image.open(paths[img_idx]), gr.Button(value="Previous Image", interactive=True), next_btn, f"{img_idx+1}/{len(paths)} : {paths[img_idx].split('/')[-1]}"

def prev_img():
    global img_idx

    if img_idx > 1:
        img_idx -= 1
        prev_btn = gr.Button(value="Previous Image")  
    elif img_idx == 1:
        img_idx -= 1
        prev_btn = gr.Button(value="No more Image", interactive=False)
    else: #first image; func should be disabled
        prev_btn = gr.Button(value="No more images, IMP", interactive=False)

    return Image.open(paths[img_idx]), prev_btn, gr.Button(value="Next Image",interactive=True), f"{img_idx+1}/{len(paths)} : {paths[img_idx].split('/')[-1]}"
    
def does_need_text(task):
    if task == 'det' or task == 'seg':
        return gr.Textbox(visible=True), gr.Image(source='upload', type="pil", value=Image.open(paths[img_idx]), tool=None, show_label=False, label="Input Image")
    elif task == 'automask':
        return gr.Textbox(value="",visible=False), gr.Image(source='upload', type="pil", value=Image.open(paths[img_idx]), tool=None, show_label=False, label="Input Image")
    else:
        return gr.Textbox(value="",visible=False), gr.Image(source='upload', type="pil", value=Image.open(paths[img_idx]), tool="sketch", show_label=False, label="Input Image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("-i","--img", type=str, help="path to the image")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument('--port', type=int, default=7589, help='port to run the server')
    parser.add_argument('--no-gradio-queue', action="store_true", help='disable gradio queue')
    args = parser.parse_args()

    print(args)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ignore subfolder and raw image
    # EXT_IGNORE = ['.dng','.json','.raw','.cr2','.cr3'] # maybe more
    EXT_IMG_ALLOW = ['.jpg','.jpeg','.png','.bmp','.tiff','.tif','.gif', '.heic', '.heif']
    paths = [args.img] if os.path.isfile(args.img) else sorted([os.path.join(args.img, f) for f in os.listdir(args.img) if os.path.isfile(os.path.join(args.img, f)) and f.lower().endswith(tuple(EXT_IMG_ALLOW))])
    print(f"Found {len(paths)}")

    # try:
    #     groundingdino_model = load_model(config_file, dino_ckpt, device=device)
    # except Exception as e:
    #     print(f"Error loading Grounding DINO model: {e}")
    
    # try:
    #     sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    #     sam.to(device=device)
    #     sam_predictor = SamPredictor(sam)
    #     sam_automask_generator = SamAutomaticMaskGenerator(sam)
    # except Exception as e:
    #     print(f"Error loading SAM model: {e}")
    
    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    with block:
        with gr.Row():
            with gr.Column():
                gr.Label(value="Model parameters", show_label=False, color='grey')
                with gr.Row(equal_height=False):
                    dino_path = gr.Text(label="Path to GroundingDINO checkpoint : ", value="checkpoints/groundingdino_swint_ogc.pth", placeholder="path to Grounding DINO checkpoint")
                    with gr.Row(equal_height=True):
                        sam_path = gr.Text(label="Path to SAM checkpoint :", value="checkpoints/sam_vit_h_4b8939.pth", placeholder="path to SAM checkpoint")
                        with gr.Column():
                            sam_version = gr.Dropdown(["vit_h", "vit_l", "vit_b"], value="vit_h", label="SAM ViT version")
                            use_sam_hq = gr.Checkbox(label="Use SAM-HQ", value=False)
                with gr.Row():
                    load_ckpts = gr.Button(variant="primary", size="lg", value="Load DINO and SAM",  container=True)
        
        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Label(value="Input image", show_label=False, color='grey')
                with gr.Group():
                    input_image = gr.Image(source='upload', type="pil", value=Image.open(paths[img_idx]), tool="sketch", show_label=False, label="Input Image")
                    count_img = gr.Label(f"{img_idx+1}/{len(paths)}: {paths[img_idx].split('/')[-1]}", show_label=False, container=False, scale=0)
                    with gr.Row(variant="panel",equal_height=True):
                        prev_button = gr.Button(value="No more images", interactive=False)
                        next_button = gr.Button(value="Next Image")

                with gr.Row():
                    task_type = gr.Dropdown([("Mask w/ interactive SAM","scribble"), ("Mask w/ automatic SAM","automask"), ("Box w/ DINO","det"), ("Box and mask w/ DINO + SAM", "seg")], value="scribble", label="Task type") #'automatic' disabled need blip and transformers
                    text_prompt = gr.Textbox(label="Text Prompt", visible=False)
                with gr.Row():
                    run_button = gr.Button(
                        label="Run", 
                        interactive=(groundingdino_model is not None and sam_predictor is not None and sam_automask_generator is not None), 
                        variant="stop")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                    )
                    scribble_mode = gr.Dropdown(["merge", "split"], value="split", label="scribble_mode")
            
            with gr.Column():
                gr.Label(value="Outputs", show_label=False, color='grey')
                gallery = Gallery(show_download_button=True, show_label=False, preview=True, object_fit="scale-down")

        load_ckpts.click(fn=load_ckpt, inputs=[dino_path, sam_path, use_sam_hq, sam_version],outputs=[dino_path, sam_path, use_sam_hq, sam_version, run_button])
        prev_button.click(fn=prev_img, outputs=[input_image, prev_button, next_button, count_img])
        next_button.click(fn=next_img, outputs=[input_image, prev_button, next_button, count_img])
        run_button.click(fn=run_grounded_sam, inputs=[input_image, text_prompt, task_type, box_threshold, text_threshold, gallery, scribble_mode], outputs=gallery)
        task_type.change(fn=does_need_text, inputs=[task_type], outputs=[text_prompt,input_image])

    block.queue(concurrency_count=100)
    serv_ip = '0.0.0.0'

    block.launch(
        server_name=serv_ip, 
        server_port=args.port, 
        debug=args.debug, 
        show_error=True,
        share=args.share, 
        inbrowser=True)
    