import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# real content of the images
true_content = [list('TODOLIST'), 
                list('1MAKEATODOLIST'),
                list('2CHECKOFFTHEFIRST'),
                list('THINGONTODOLIST'),
                list('3REALIZEYOUHAVEALREADY'),
                list('COMPLETED2THINGS'),
                list('4REWARDYOURSELFWITH'),
                list('ANAP'),
                list('ABCDEFG'),
                list('HIJKLMN'),
                list('OPQRSTU'),
                list('VWXYZ'),
                list('1234567890'),
                list('HAIKUSAREEASY'),
                list('BUTSOMETIMESTHEYDONTMAKESENSE'),
                list('REFRIGERATOR'),
                list('DEEPLEARNING'),
                list('DEEPERLEARNING'),
                list('DEEPESTLEARNING')]

count = 0        # count for the images

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.figure()
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.    
    row_boxes = np.array(bboxes)[:, 0].reshape(-1, 1)
    # print(row_boxes)
    rows = []
    current_row = []
	# clustering    
    for i in range(row_boxes.shape[0]-1):
        current_row.append(bboxes[i])
        if row_boxes[i+1] - row_boxes[i] > 100:
            rows.append(current_row)
            current_row = []     
    current_row.append(bboxes[-1])
    rows.append(current_row)
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    total_char = 0
    correct = 0
    for eachRow in rows:
		# sort each row by center
        sorted_row = sorted(eachRow, key = lambda x: (x[1] + x[3])//2 )
        row_content = []
        for bbox in sorted_row:
            # get each letter cropped
            crop_image = bw[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            # padding
            pad_factor  = max(crop_image.shape[0], crop_image.shape[1])
            pad_y = int((pad_factor - crop_image.shape[0])//2 + pad_factor//10)
            pad_x = int((pad_factor - crop_image.shape[1])//2 + pad_factor//10)
            crop_padding = np.pad(crop_image, [(pad_y, pad_y),(pad_x, pad_x)], 'constant',constant_values=(1, 1))     
            # resize               
            crop_final = skimage.transform.resize(crop_padding, (32, 32)) 
            # according to the plot, the resized images are pretty unclear so I am using erosion here to emphasize the letter      
            crop_final = skimage.morphology.erosion(crop_final,skimage.morphology.square(3))
            # flatten
            row_content.append(crop_final.T.flatten())
   
        row_content = np.array(row_content)
        # forward
        h1 = forward(row_content, params,'layer1')
        probs = forward(h1, params, 'output', softmax)
        predicted_values = np.argmax(probs, axis=1)
        predicted_content = [letters[i] for i in predicted_values]
        
        print(''.join(predicted_content))
        
        for j, eachChar in enumerate(true_content[count]):
            total_char +=1
            if j >= len(predicted_content):
                break
            if eachChar == predicted_content[j]:
                correct+=1
        
        count+=1
    print(correct)
    print('Accuracy: ', correct/total_char)
        
    
