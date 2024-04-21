# Import the library mitsuba using the alias 'mi'
import mitsuba as mi

# Set the variant of the renderer
mi.set_variant('cuda_ad_rgb')

# Import other libraries
from tqdm import tqdm
import os
from utils import *
import argparse
from scipy import ndimage
from PIL import Image

# Serrano dataset: 100
# Our dataset: 6
MASK_CHANNEL = 6

parser = argparse.ArgumentParser(description='Exr to png.')

# Path to exr files, a folder or a patern (list of files), e.g., example_dir/*.exr
parser.add_argument('-i', '--input_path', type=str, nargs='+', required=True)   
parser.add_argument('-l', '--file_list', type=str, required=True)  

# Output path
parser.add_argument('-o', '--output_path', type=str, required=True) 

# Crop size
parser.add_argument('-s', '--size', type=int, default=256) 

# Exposure for the gamma-curve tonemapper
parser.add_argument('-t', '--tm_fstop', type=float, default=1.5) 

# Apply a median filter to remove black pixels
parser.add_argument('-m', '--median_filter', type=str, default="False") 

# Remove background
parser.add_argument('-a', '--mask', type=str, default="False")  

args = parser.parse_args()
args.file_list = args.file_list == 'True'
args.mask = args.mask == 'True'
args.median_filter = args.median_filter == 'True'

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# Get files' names
if args.file_list:

    exr_files = args.input_path
else:
   
    args.input_path = args.input_path[0]
    exr_files = os.listdir(args.input_path)
    exr_files.sort()


for id_scene in tqdm(range(len(exr_files))): 

    exr_file = exr_files[id_scene]

    # Parse files' names
    if args.file_list:
        bmp = mi.Bitmap(exr_file)
        exr_file = exr_file.split('/')[-1]
    else:
        bmp = mi.Bitmap(os.path.join(args.input_path, exr_file))

    bmp_np = np.array(bmp)

    # Skipp if the file exists
    if os.path.exists(os.path.join(args.output_path, exr_file[:-3] + 'png')):
        continue

    # Select the rgb channels 
    # Serrano dataset: 3:6 (3,4,5)
    # Analytical dataset: :3 (0,1,2)
    img_np = np.copy(bmp_np[:,:,:3]) 

    # Apply tonemapper
    tonemaped_img = (tonemapping(img_np, args.tm_fstop) * 255).astype(np.uint8)
    number_nans = np.count_nonzero(np.isnan(tonemaped_img))

    # Apply a median filter to remove black pixels
    if args.median_filter and number_nans > 0:

        tonemaped_img[:,:,0] = ndimage.median_filter(tonemaped_img[:,:,0], size=args.median_filter)
        tonemaped_img[:,:,1] = ndimage.median_filter(tonemaped_img[:,:,1], size=args.median_filter)
        tonemaped_img[:,:,2] = ndimage.median_filter(tonemaped_img[:,:,2], size=args.median_filter)

    # Remove background
    if args.mask:
        mask = np.copy((bmp_np[:,:,MASK_CHANNEL] != 0 ).astype(np.uint8))
        tonemaped_img[:,:,0]  = tonemaped_img[:,:,0]  * mask
        tonemaped_img[:,:,1]  = tonemaped_img[:,:,1]  * mask
        tonemaped_img[:,:,2]  = tonemaped_img[:,:,2]  * mask

    tonemaped_img = Image.fromarray(tonemaped_img)
    tonemaped_img = tonemaped_img.resize((args.size, args.size))
    tonemaped_img.save(os.path.join(args.output_path, exr_file[:-3] + 'png'))
    