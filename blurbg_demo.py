from argparse import ArgumentParser
import os
from tqdm import tqdm

from PIL import Image
import numpy as np
from cv2 import threshold, THRESH_BINARY

from blur_detector import detectBlur

IMG_EXT = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.tif', '.webp']

def process_file(input_folder, file, output_folder, thr_min: float = 0.1):
    
    if file.lower().endswith(tuple(IMG_EXT)):
        image_pil = Image.open(os.path.join(input_folder, file))
        file, ext = os.path.splitext(file)
        output_file = os.path.join(output_folder, file + '.png')

        image_pil = image_pil.convert("L")
        image = np.array(image_pil)
        blur = detectBlur(image, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=False)

        # Post edit
        _, th1 = threshold(blur, thr_min, 1, THRESH_BINARY)

        # Save th1
        th1 = (th1 * 255).astype('uint8')
        th1 = Image.fromarray(th1)
        th1.save(output_file)

    else:
        print(f"Unsupported file type: {file}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Remove Image Background')
    parser.add_argument('-i','--input', help='Input image file')
    parser.add_argument('-o','--output', help='Output image file')
    parser.add_argument('--threshold', type=float, help='Threshold for blur detection')
    args = parser.parse_args()

    assert(args.input != args.output)
    assert(os.path.exists(args.input))
    
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input): # Process file
        parent_folder = os.path.dirname(args.input)
        process_file(parent_folder, args.input, args.output, args.threshold)
    elif os.path.isdir(args.input): # Process folder
        for file in tqdm(sorted(os.listdir(args.input))):
            process_file(args.input, file, args.output, args.threshold)
    else:
        print(f"Unsupported input type: {args.input}")


