
# coding: utf-8

# In[1]:


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


# In[2]:


batch_size = 15
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10
epochs = 24
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


# fig = plt.figure(figsize=(8,3))
# for i in range(num_classes):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     idx = np.where(y_train[:]==i)[0]
#     features_idx = x_train[idx,::]
#     img_num = np.random.randint(features_idx.shape[0])
#     im = features_idx[img_num,::]
#     ax.set_title(class_names[i])
#     plt.imshow(im)
# plt.show()


# In[4]:


sampling = 50000
x_train = x_train.astype('float32')[:sampling]
y_train = np_utils.to_categorical(y_train[:sampling], num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


# In[11]:


def custom_gabor(shape, dtype=None):
    orientation_spread = np.linspace(0, 4, 8) / 4. * np.pi
    scales = np.linspace(0.1, 0.4, 6)
    real_kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            real_kernel = cv2.getGaborKernel((3, 3), 3, orientation, scale, 1, 0)
            real_kernel = np.delete(np.delete(real_kernel, -1, 0), -1, 1)
            real_kernels.append(real_kernel)
    real_kernels = np.array([real_kernels, real_kernels, real_kernels])
    real_kernels = np.einsum('hijk->jkhi', real_kernels)
    print(real_kernels.shape)

    real_kernels = K.variable(real_kernels)
#     print(real_kernels.shape)
    random = K.random_normal(shape, dtype=dtype)
    print('here')
    print(random)
    print(random.shape)
    return real_kernels


# In[12]:


def base_model():
    model = Sequential()
    model.add(Conv2D(48, (3, 3), padding='same', kernel_initializer=custom_gabor, data_format='channels_last', input_shape=x_train.shape[1:]))
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


# In[ ]:


# print(x_train.shape[1:])

cnn_n = base_model()
cnn_n.layers[0].tranable = False
cnn_n.summary()
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


# # Evaluation

# In[ ]:


score = cnn_n.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# cnn_n.layers[0].get_weights()


# In[ ]:


# # Plots for training and testing process: loss and accuracy

# plt.figure(0)
# plt.plot(cnn.history['acc'],'r')
# plt.plot(cnn.history['val_acc'],'g')
# plt.xticks(np.arange(0, epochs, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy vs Validation Accuracy")
# plt.legend(['train','validation'])


# plt.figure(1)
# plt.plot(cnn.history['loss'],'r')
# plt.plot(cnn.history['val_loss'],'g')
# plt.xticks(np.arange(0, epochs, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Validation Loss")
# plt.legend(['train','validation'])


# plt.show()
