import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from scipy.stats import skew
from tqdm import tqdm

fig = plt.figure(figsize=(8,4))
ax = []
parser = argparse.ArgumentParser(description='Skew Computing.')
parser.add_argument('-hi', '--histogram', choices=('True','False'))   
parser.add_argument('-m', '--mask', choices=('True','False')) 

# Path to folder with subfolders containing png files
parser.add_argument('-p', '--path', type=str, required=True) 
parser.add_argument('-r', '--red_lines', choices=('True','False'))

""" Apply a tronsform to the histogram: 
        e: equalization
        f: compute in frequency space
        n: nothing
"""
parser.add_argument('-t', '--transform', choices=('e','f','n'))
parser.add_argument('-i', '--inverse', choices=('True','False'))

""" Select the statistic to compute: 
        s: skewness
        k: kurtosis
"""
parser.add_argument('-s', '--moment', choices=('s','k'))
parser.add_argument('-d', '--deviation', type=int, required=True)
args = parser.parse_args()
args.red_lines = args.red_lines == 'True'
args.mask = args.mask == 'True'
args.histogram = args.histogram == 'True'

args.inverse = args.inverse == 'True'
metrics = open('statistics.txt', 'w')

if args.transform == 'f':
    args.mask = True

if args.histogram:

    if args.transform == 'e':
        if not os.path.exists('{}_eq_hist'.format(args.path[:-1])):
            os.mkdir('{}_eq_hist'.format(args.path[:-1]))
    elif args.transform == 'f':
        if not os.path.exists('{}_spec'.format(args.path[:-1])):
            os.mkdir('{}_spec'.format(args.path[:-1]))
    
    if not os.path.exists('{}_hist'.format(args.path[:-1])):
        os.mkdir('{}_hist'.format(args.path[:-1]))

if not os.path.exists('{}_grey'.format(args.path[:-1])):
    os.mkdir('{}_grey'.format(args.path[:-1]))
else:
    mask = 0

# Get statistics as integers
def get_label(skewness,hist):

    for i in range(1,hist.shape[0]):

        if skewness <= hist[i] and skewness > hist[i-1]:
            return i
    return 1

# Get statistics as floats
def get_label_continuous(skewness,hist):

    min_value = np.min(hist)
    max_value = np.max(hist)
    if min_value < 0:
        
        skewness += np.abs(min_value)
        label = skewness /  (np.abs(max_value) + np.abs(min_value))
        label = label * 6

    else:

        label = skewness /  np.abs(max_value)
        label = label * 6

    return label + 1

# Compute the image statistics
def compute_moment(path, id, mask_name):

    if args.mask:
        
        mask = cv.imread('masks/{}'.format(mask_name) , cv.IMREAD_GRAYSCALE)

    image = cv.imread(path)
    gray = cv.cvtColor(image[:,:,:3], cv.COLOR_BGR2GRAY)
    cv.imwrite('{}_grey/{}_{}.png'.format(args.path[:-1],args.path[:-1], id), gray)

    N,M = gray.shape
    total_pixels = N * M

    # Apply a mask
    if args.mask:

        gray  = gray.reshape((N * M)).astype(np.int64)
        mask  = mask.reshape((N * M))
        gray = gray[mask != 0] 
        #print('{} peak: '.format(id) , np.where(gray == 255)[0].shape[0])
        #gray = np.delete(gray, np.where(gray == 255))
        N_pixels = gray.shape[0]

    else:
        N_pixels = N * M
        gray  = gray.reshape((N_pixels))

    
    if args.histogram:

        if args.transform == 'e':

            #flatten image array and calculate histogram via binning
            histogram_array = np.bincount(gray, minlength=256)
            #normalize
            num_pixels = np.sum(histogram_array)
            histogram_array = histogram_array/num_pixels
            #normalized cumulative histogram
            chistogram_array = np.cumsum(histogram_array)
            transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
            # flatten image array into 1D list
            img_list = list(gray)
            # transform pixel values to equalize
            eq_img_list = [transform_map[p] for p in img_list]
            mean = np.mean(eq_img_list)
            std = np.std(eq_img_list)
            plt.figure()
            plt.axvline(x = mean, color = 'r')

            if args.deviation > 0:
                plt.axvline(x = mean - args.deviation *std, color = 'g')
                plt.axvline(x = mean + args.deviation *std, color = 'g')

            plt.hist(eq_img_list, bins=256)
            plt.savefig('{}_eq_hist/{}_{}.png'.format(args.path[:-1],args.path[:-1], id))
            plt.close() 

        elif args.transform == 'f':

            f = np.fft.fft2(cv.cvtColor(image[:,:,:3], cv.COLOR_BGR2GRAY))
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))

            plt.imshow(magnitude_spectrum, cmap = 'gray')
            plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            plt.savefig('{}_spec/{}_{}.png'.format(args.path[:-1],args.path[:-1], id))
            plt.close() 
            magnitude_spectrum = magnitude_spectrum.reshape((total_pixels))
            mean = np.mean(magnitude_spectrum)
            std = np.std(magnitude_spectrum)
            plt.figure()
            plt.axvline(x = mean, color = 'r')

            if args.deviation > 0:
                plt.axvline(x = mean - args.deviation * std, color = 'g')
                plt.axvline(x = mean + args.deviation * std, color = 'g')

            plt.hist(magnitude_spectrum, bins=256)
            plt.savefig('{}_hist/{}_{}.png'.format(args.path[:-1],args.path[:-1], id))
            plt.close() 

        else:
            
            mean = np.mean(gray)
            std = np.std(gray)
            plt.figure()
            plt.axvline(x = mean, color = 'r')

            if args.deviation > 0:
                plt.axvline(x = mean - args.deviation * std, color = 'g')
                plt.axvline(x = mean + args.deviation * std, color = 'g')

            plt.hist(gray, bins=256)
            plt.savefig('{}_hist/{}_{}.png'.format(args.path[:-1], args.path.split('/')[2], id))
            plt.close() 

    # Save the histogram
    PIL_img = Image.open(path)
    PIL_img = PIL_img.convert('RGB')
    PIL_img = PIL_img.resize((512, 512))
    ax.append(fig.add_subplot(5, 10, id + 1))
    plt.axis('off')
    plt.imshow(np.array(PIL_img))
    metrics.write('Img {}:\n'.format(id))
    metrics.write('    Izq: {}\n'.format(np.where(gray < mean - args.deviation *std)[0].shape[0]))
    metrics.write('    Der: {}\n'.format(np.where(gray >= mean +  args.deviation *std)[0].shape[0]))
    metrics.write('    Var: {}\n'.format(std * std))
    metrics.write('    Std: {}\n'.format(std))
    metrics.write('    Mean: {}\n'.format(mean))
    
    if args.transform == 'f':
        gray = magnitude_spectrum
        N_pixels = total_pixels

    # Compute the selected statistics
    if args.moment == 's':
        
        moment = np.sum((gray - mean) ** 3) /  (std ** 3 * N_pixels)
        metrics.write('    Skewness: {}\n\n'.format(moment))
    elif args.moment == 'k':

        moment = np.sum((gray - mean) ** 4) /  (std ** 4 * N_pixels)
        metrics.write('    Kurtosis: {}\n\n'.format(moment))
    
    if id % 10 == 9: 
        metrics.write('-----------------------\n')

    if args.inverse: 
        return -moment
    else:
        return moment

files = os.listdir(args.path)
files.sort()

num_imgs = len(files)
skewness_values = np.ones(num_imgs)
labels = np.ones(num_imgs)

for image_id in tqdm(range(num_imgs)):
    image_name = files[image_id]
    skewness_values[image_id] = compute_moment(args.path + image_name, image_id, image_name)
metrics.close()
hist, bin_edges = np.histogram(skewness_values, bins=7)

labels_file = open('{}_labels.txt'.format(args.path[:-1]), 'w')
for image_id in range(num_imgs):

    image_name = files[image_id]
    # To obtain image statistics with decimal part
    # labels[image_id] = get_label_continuous(skewness_values[image_id],bin_edges)
    labels[image_id] = get_label(skewness_values[image_id],bin_edges)
    labels_file.write('{}, {}\n'.format(image_name, labels[image_id]))
    ax[image_id].title.set_text('{}'.format(int(labels[image_id])))

labels_file.close()

y = np.linspace(1,num_imgs,num_imgs)
plt.figure()

if args.red_lines:
    for edge in range(0,bin_edges.shape[0]):
        plt.axhline(y = bin_edges[edge], color = 'r')

plt.plot(y, skewness_values,'o')

# Show statistics for each image
plt.figure()
plt.plot(y, labels,'o')
plt.show()

