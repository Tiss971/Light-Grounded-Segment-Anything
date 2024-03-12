from rembg import remove
from argparse import ArgumentParser
from PIL import Image
import os
from tqdm import tqdm

IMG_EXT = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.tif', '.webp']

def process_file(input_folder, file, output_folder, remove_bg_mode, mask_bg, post_process_mask):
    
    if file.lower().endswith(tuple(IMG_EXT)):
        image_pil = Image.open(os.path.join(input_folder, file))
        file, ext = os.path.splitext(file)
        output_file = os.path.join(output_folder, file+'.png')

        bgcolor=(0, 0, 0, 255) if mask_bg else None

        # Remove background
        if remove_bg_mode == 'alpha_matting':
            front = remove(image_pil, alpha_matting=True, post_process_mask=post_process_mask, bgcolor=bgcolor)
        elif remove_bg_mode == 'only_mask':
            front = remove(image_pil, alpha_matting=False, post_process_mask=post_process_mask, bgcolor=bgcolor)
        else:
            front = remove(image_pil, alpha_matting=False, post_process_mask=post_process_mask, bgcolor=bgcolor)

        # Post edit
        if mask_bg:
            front = front.convert('L')
            front = front.point(lambda p: p > 0 and 255)

        front.save(output_file)
    else:
        print(f"Unsupported file type: {file}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Remove Image Background')
    parser.add_argument('-i','--input', help='Input image file')
    parser.add_argument('-o','--output', help='Output image file')
    parser.add_argument('--alpha-matting', action='store_true', help='Use alpha matting')
    parser.add_argument('--mask-bg', action='store_true', help='Mask background')
    parser.add_argument('--post-process-mask', action='store_true', help='Post process mask')
    args = parser.parse_args()

    assert(args.input != args.output)
    assert(os.path.exists(args.input))
    
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input): # Process file
        parent_folder = os.path.dirname(args.input)
        process_file(parent_folder, args.input, args.output, args.alpha_matting, args.mask_bg, args.post_process_mask)
    elif os.path.isdir(args.input): # Process folder
        for file in tqdm(sorted(os.listdir(args.input))):
            process_file(args.input, file, args.output, args.alpha_matting, args.mask_bg, args.post_process_mask)
    else:
        print(f"Unsupported input type: {args.input}")


