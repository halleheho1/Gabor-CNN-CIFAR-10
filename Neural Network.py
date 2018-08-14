
# coding: utf-8

# In[136]:


import time
import matplotlib.pyplot as plt
import numpy as np
import keras
from skimage.filters import gabor_kernel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")
from skimage.color import rgb2gray
from scipy import ndimage as ndi
 
# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp
 
# Loading the CIFAR-10 datasets
from keras.datasets import cifar10
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[137]:


batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10
epochs = 24
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[138]:


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst


# In[139]:


def add_dimension(data):
    data = np.array([data])
    #re arange the dimension
    print(data.shape)
    data = np.einsum('hijk->ijkh', data)
    return data


# In[140]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_selected_amount = 50000
test_selected_amount = 10000
num_classes = 10

init_y_train = y_train[:train_selected_amount]
init_y_test = y_test[:test_selected_amount]

x_train = add_dimension(grayscale(x_train[:train_selected_amount]))
x_test = add_dimension(grayscale(x_test[:test_selected_amount]))
y_train = np_utils.to_categorical(init_y_train, num_classes)
y_test = np_utils.to_categorical(init_y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


# In[141]:





# In[142]:


def custom_gabor(shape, dtype=None):
    orientation_spread = np.linspace(0, 4, 8) / 4. * np.pi
    scales = np.linspace(0.1, 0.4, 6)
    real_kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            real_kernel = cv2.getGaborKernel((3, 3), 3, orientation, scale, 1, 0)
#             real_kernel = np.delete(np.delete(real_kernel, -1, 0), -1, 1)
            real_kernels.append(real_kernel)
    real_kernels = np.array([real_kernels])
    real_kernels = np.einsum('hijk->jkhi', real_kernels)
    print(real_kernels.shape)

    real_kernels = K.variable(real_kernels)
#     print(real_kernels.shape)
    random = K.random_normal(shape, dtype=dtype)
#     print('here')
#     print(random)
#     print(random.shape)
    return real_kernels


# In[143]:


def base_model(shape):
    model = Sequential()
    model.add(Conv2D(48, (3, 3), padding='same',kernel_initializer=custom_gabor, data_format='channels_last', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # sgd = SGD(lr = 0.1, decay = 1e-6, momentum=0.9, nesterov=True)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Train model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[122]:


# cnn_n = base_model(x_train.shape[1:])
# cnn_n.summary()

# cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


# In[123]:


# score = cnn_n.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# ### K fold cross validation

# In[147]:


k = 10
scores = []
folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_train, init_y_train))
for j, (train_idx, val_idx) in enumerate(folds):
    print('fold ', j)
    x_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    x_valid_cv = x_train[val_idx]
    y_valid_cv = y_train[val_idx]
    model = base_model(x_train_cv.shape[1:])
    model.fit(x_train_cv, y_train_cv, batch_size=batch_size, epochs=epochs, validation_data=(x_valid_cv, y_valid_cv), shuffle=True)
    score = model.evaluate(x_test, y_test, verbose=0)
    scores.append(score[1] * 100)
print("average accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))


# In[ ]:


# serialize model to JSON
model_json = cnn_n.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_n.save_weights("models/model.h5")
print("Saved model to disk")

