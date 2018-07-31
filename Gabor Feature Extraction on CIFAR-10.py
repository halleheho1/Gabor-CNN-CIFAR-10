
# coding: utf-8
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

batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch

num_classes = 10
epochs = 24
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num,::]
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()


# ## Convert to Grayscale images

def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst

unique, count = np.unique(y_train[:5000], return_counts=True)
print(count)

train_selected_amount = 50000
test_selected_amount = 10000
gray_x_train = grayscale(x_train[:train_selected_amount])
gray_x_test = grayscale(x_test[:test_selected_amount])
y_train = np_utils.to_categorical(y_train[:train_selected_amount], num_classes)
y_test = np_utils.to_categorical(y_test[:test_selected_amount], num_classes)
print(y_train.shape)
print(y_test.shape)



def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    real_part = ndi.convolve(image, np.real(kernel), mode='wrap')
    imagine_part = ndi.convolve(image, np.imag(kernel), mode='wrap')
    return {'real_part': real_part, 'imagine_part': imagine_part}

def convolve(images):
    orientation_spread = np.linspace(0, 4, 8) / 4. * np.pi
    scales = np.linspace(0.1, 0.4, 6)
    real_list = []
    imagine_list = []
    kernels = []
    kernel_params = []
    for theta in orientation_spread:
        for frequency in scales:
            kernel = gabor_kernel(frequency, theta=theta)
            kernels.append(kernel)
            params = 'theta=%d, frequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            real_list.append([power(img, kernel)['real_part'] for img in images])
            imagine_list.append([power(img, kernel)['imagine_part'] for img in images])
    real_list = np.array(real_list)
    imagine_list = np.array(imagine_list)
    stacked_list = np.vstack((real_list, imagine_list))
    return stacked_list, kernels, kernel_params


results_train, kernels, kernel_params= convolve(gray_x_train)
convolved_x_test, kernels, kernel_params= convolve(gray_x_test)

convolved_results = results_train
convolved_x_test = convolved_x_test
print(convolved_results.shape)
print(convolved_x_test)

np.save('convolved_96_real_imagine_train', results_train)
np.save('convoled_96_real_imagine_test', convolved_x_test)

convolved_results = np.array(results_train)
print(convolved_results.shape)
convolved_results = np.einsum('abcd->bcda', convolved_results)
convolved_x_test = np.einsum('abcd->bcda', convolved_x_test)
print(convolved_results.shape)

def custom_gabor(shape, dtype=None):
    kernels = []
    for theta in (0, 1):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            print(type(kernel))
            kernels.append(kernel)
    kernels = np.array(kernels)
    print(kernels.shape)
    return K.variable(kernels, dtype=dtype)

def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=convolved_results.shape[1:]))
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

print(y_train.shape)
cnn_n = base_model()
print(convolved_results.shape)
print(convolved_x_test.shape)
cnn_n.summary()

cnn = cnn_n.fit(convolved_results, y_train, batch_size=batch_size, epochs=epochs, validation_data=(convolved_x_test, y_test), shuffle=True)

score = cnn_n.evaluate(convolved_x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
