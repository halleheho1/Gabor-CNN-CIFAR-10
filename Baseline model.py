
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


def add_dimension(data):
    data = np.array([data])
    data = np.einsum('hijk->ijkh', data)
    return data


# In[5]:


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    rst = add_dimension(rst)
    return rst


# In[6]:


sampling = 50000
x_train = grayscale(x_train.astype('float32')[:sampling])
y_train = np_utils.to_categorical(y_train[:sampling], num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_test = grayscale(x_test.astype('float32'))
x_train  /= 255
x_test /= 255


# In[7]:


print(x_train.shape)


# In[8]:


def base_model():
    model = Sequential()
    model.add(Conv2D(48, (3, 3), padding='same', data_format='channels_last', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3)))
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


# In[9]:


# print(x_train.shape[1:])

model = base_model()
model.summary()
cnn = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


# # Evaluation

# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


model_json = model.to_json()
with open("models/baseline.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("models/baseline.h5")
print("Saved model to disk")


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

