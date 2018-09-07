import time
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

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10
from keras.models import model_from_json
import cv2

def custom_gabor(shape, dtype=None):
    pi = np.pi
    orientation_spread = np.array([0, pi/4, pi/2, pi*3/4, pi, pi*5/4, pi*3/2, 2*pi])
    scales = np.linspace(2, 3, 6)
    real_kernels = []
    img_kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            real_kernel = cv2.getGaborKernel((5, 5), 1, orientation, scale, 1, 0)
            imaginary_kernel = cv2.getGaborKernel((5, 5), 1, orientation, scale, 1, np.pi / 2)
#             real_kernel = np.delete(np.delete(real_kernel, -1, 0), -1, 1)
            real_kernels.append(real_kernel)
            img_kernels.append(imaginary_kernel)
    stacked_list = np.vstack((real_kernels, img_kernels))
    stacked_list = np.array([stacked_list])
    stacked_list = np.einsum('hijk->jkhi', stacked_list)
    stacked_list = K.variable(stacked_list)
    random = K.random_normal(shape, dtype=dtype)
    return stacked_list

def preprocessing_pipeline():
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
    return x_train, x_test, y_train, y_test

def base_model(shape, is_baseline):
    num_classes = 10
    model = Sequential()
    if is_baseline:
        model.add(Conv2D(48, (3, 3), padding='same', data_format='channels_last', input_shape=shape))
    else:
        model.add(Conv2D(96, (3, 3), padding='same',kernel_initializer=custom_gabor, data_format='channels_last', input_shape=shape))
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
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def load_model_from_weight(weight, is_baseline):
    x_train, x_test, y_train, y_test = preprocessing_pipeline()
    model = base_model(x_train.shape[1:], is_baseline)
    model.load_weights(weight)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst

def add_dimension(data):
    data = np.array([data])
    #re arange the dimension
    data = np.einsum('hijk->ijkh', data)
    return data

def evaluate(model):
    x_train, x_test, y_train, y_test = preprocessing_pipeline()
    predicted = model.predict(x_test)
    predicted_reverse = predicted.argmax(1)
    accuracy = model.evaluate(x_test, y_test, verbose=0)
    return predicted_reverse, accuracy
