import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches
import time
import numpy as np
import string
from PIL import Image


from q4 import *

letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])


def get_data_from_image(img_path):
    im1 = skimage.img_as_float(skimage.io.imread(img_path))
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
    rows_content = []
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
            crop_final = skimage.transform.resize(crop_padding, (28, 28))
            # according to the plot, the resized images are pretty unclear so I am using erosion here to emphasize the letter      
            crop_final = skimage.morphology.erosion(crop_final,skimage.morphology.square(3))
            # flatten
            row_content.append(crop_final.T)
        rows_content.append(row_content)
    return rows_content


def detect(input_data):
    tss = []
    for item in input_data:
        # print(item.shape)
        # item = item.reshape(28, 28)
        ts = transform(Image.fromarray(item.astype('uint8'), 'L'))
        tss.append(ts)
    x_ts = torch.stack(tss, dim=0)

    # get the inputs
    inputs = x_ts.to(device)

    # get output
    y_pred = model(inputs)

    predicted = torch.max(y_pred, 1)[1]

    result = predicted.cpu().numpy()
    row_s = ''
    for i in range(result.shape[0]):
        row_s += letters[int(result[i])]

    print(row_s)


def detect_for_images():
    for img in os.listdir('../images'):
        img_path = os.path.join('../images',img)
        input_data = get_data_from_image(img_path)
        for row_data in input_data:
            detect(row_data)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    max_iters = 20
    print("here")
    # pick a batch size, learning rate
    batch_size = 64
    learning_rate = 0.01

    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

    trainset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True,
                                           download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False,
                                          download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    model = torchvision.models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 36)
    model = model.to(device)

    train_loss = []
    train_acc = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    since = time.time()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for epoch in range(max_iters):
        print('Epoch {}/{}'.format(epoch, max_iters - 1))
        print('-' * 10)

        scheduler.step()
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainset)
        epoch_acc = running_corrects.double() / len(trainset)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    plt.figure('accuracy')
    plt.plot(range(max_iters), train_acc, color='g')
    plt.legend(['train accuracy'])
    plt.show()

    plt.figure('loss')
    plt.plot(range(max_iters), train_loss, color='g')
    plt.legend(['train loss'])
    plt.show()

    print('Train accuracy: {}'.format(train_acc[-1]))

    torch.save(model.state_dict(), "q7_1_4_model_parameter.pkl")

    # checkpoint = torch.load('q7_1_4_model_parameter.pkl')
    # model.load_state_dict(checkpoint)

    # # run on test data
    # model.eval()  # Set model to evaluation mode
    # since_test = time.time()
    # # Iterate over data.
    # for inputs in test_loader:
    #     inputs = inputs.to(device)
    #     outputs = model(inputs)
    #     _, preds = torch.max(outputs, 1)
    #
    # time_elapsed = time.time() - since_test
    # print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('###### Test ends here ######')
    #
    # print('Test accuracy: {}'.format(test_acc))
    detect_for_images()
