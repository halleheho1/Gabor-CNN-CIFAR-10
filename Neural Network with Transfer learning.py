
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
from keras_sequential_ascii import sequential_model_to_ascii_printout
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
from keras.models import model_from_json


# In[2]:


json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")


# ### Freeze first layer

# In[3]:


loaded_model.layers[0].trainable = False


# In[4]:


# loaded_model.summary(line_length=150)
# loaded_model.pop()
# len(loaded_model.layers)


# In[5]:


batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10
epochs = 24
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[8]:


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = np.load('convolved_x_train_cv2.npy')
x_test = np.load('convolved_x_test_cv2.npy')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


# In[7]:


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
cnn = loaded_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)


# In[ ]:


score = cnn_n.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# serialize model to JSON
model_json = cnn_n.to_json()
with open("models/transferred_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn_n.save_weights("models/transferred_model.h5")
print("Saved model to disk")

