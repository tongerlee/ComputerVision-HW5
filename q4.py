import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    img_in = skimage.img_as_float(image)
    # blur: noise + denoise
    # sigma = 0.155
    # noisy = skimage.util.random_noise(img_in)
    denoise_img = skimage.restoration.denoise_bilateral(img_in,multichannel=True)
    # greyscale
    gray = skimage.color.rgb2gray(denoise_img)
    # threshold
    th = skimage.filters.threshold_otsu(gray)
    # morphology
    #open_img = skimage.morphology.opening(gray <= th)
    closed_img = skimage.morphology.closing(gray <= th, skimage.morphology.square(9))
   
    # label
    label_image = skimage.measure.label(closed_img)
    regions = skimage.measure.regionprops(label_image)
    
    # skip small boxes
    avg_size = sum([eachBox.area for eachBox in regions])/len(regions)
    bboxes = [eachBox.bbox for eachBox in regions if eachBox.area > avg_size/3]

    bw = 1.0 - closed_img
    
    return bboxes, bw
