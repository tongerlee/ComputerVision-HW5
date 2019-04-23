import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import matplotlib
import copy

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
# a single hidden layer with 64 hidden units and train for at least 30 epochs
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%
loss_list = []
acc_list = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta1 = copy.deepcopy(probs)
        delta1[np.arange(probs.shape[0]),yb.argmax(axis=1)] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        # Implement backwards!
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']

    total_acc /= len(batches)
    total_loss /= len(batches)
    acc_list.append(total_acc)
    loss_list.append(total_loss)
    # training loop can be exactly the same as q2!
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))


# run on validation set and report accuracy! should be above 75%
valid_acc = None
batches = get_random_batches(valid_x,valid_y,batch_size)
valid_acc_list = []
valid_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
		# forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        loss, valid_acc = compute_loss_and_acc(yb, probs)
        total_loss+= loss
        total_acc += valid_acc
    total_acc = total_acc/batch_num
    total_loss = total_loss/batch_num
    valid_acc_list.append(total_acc)
    valid_loss_list.append(total_loss)
    
print('Validation accuracy: ',valid_acc)

# loss plot
plt.plot(np.arange(max_iters), loss_list, color = 'b')
plt.plot(np.arange(max_iters), valid_loss_list, color = 'r')
plt.legend(['train', 'validation'])
plt.xlabel('Epoches')
plt.ylabel('Training Loss over epoches')
plt.show()

# accuracy plot
plt.plot(np.arange(max_iters), acc_list, color = 'b')
plt.plot(np.arange(max_iters), valid_acc_list, color = 'r')
plt.legend(['train', 'validation'])
plt.xlabel('Epoches')
plt.ylabel('Training Accuracy over epoches')
plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
        
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)

# First Layer weights after training
weights = saved_params['Wlayer1']

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8,8))
for i in range(hidden_size):
    grid[i].imshow(saved_params['Wlayer1'][:, i].reshape(32, 32))
plt.show()

# Initial weights
initialize_weights(1024, 64, saved_params, 'initial')

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8,8))
for i in range(64):
    grid[i].imshow(saved_params['Winitial'][:, i].reshape(32, 32))
plt.show()


# Q3.1.4
with open('q3_weights.pickle', 'rb') as handle:
   saved_params = pickle.load(handle)

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# forward
h1 = forward(valid_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)

true_values = np.argmax(valid_y, axis=1)
predicted_values = np.argmax(probs, axis=1)

for i in range(true_values.shape[0]):
    confusion_matrix[true_values[i], predicted_values[i]] += 1

import string
plt.figure()
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.savefig('confusion_matrix.png')
plt.show()
