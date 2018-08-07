
# coding: utf-8

# In[10]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from keras.datasets import cifar10
from keras.utils import np_utils


# In[90]:


def create_kernel():
    orientation_spread = np.linspace(0, 4, 8) / 4. * np.pi
    scales = np.linspace(0.1, 0.4, 6)
    kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            kernels.append(cv2.getGaborKernel((31, 31), 3, orientation, scale, 1))
    return np.array(kernels)


# In[91]:


kernels = create_kernel()


# In[92]:


# fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(15, 20))
# plt.gray()
# fig.suptitle('Gabor kernels', fontsize=12)
# idx = 0
# for row in range(0, 8):
#     for col in range(0, 6):
#         cell = axes[row][col]
#         cell.imshow(kernels[idx])
#         idx += 1


# In[77]:


kernels.shape


# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[ ]:


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst


# In[15]:


def apply_gabor(img, kernel):
    return cv2.filter2D(img, cv2.CV_32F, kernel)


# In[30]:


def convolve(images):
    orientation_spread = np.linspace(0, 4, 8) / 4. * np.pi
    scales = np.linspace(0.1, 0.4, 6)
    real_list = []
    imagine_list = []
    kernels = []
    kernel_params = []
    iteration = 1
    for theta in orientation_spread:
        for frequency in scales:
            real_kernel = cv2.getGaborKernel((31, 31), 3, theta, frequency, 1, 0)
            imaginary_kernel = cv2.getGaborKernel((31, 31), 3, theta, frequency, 1, np.pi / 2)
            params = 'theta=%d, frequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            real_list.append([apply_gabor(img, real_kernel) for img in images])
            imagine_list.append([apply_gabor(img, imaginary_kernel) for img in images])
            print('finished iter:' + str(iteration))
            iteration += 1
    real_list = np.array(real_list)
    imagine_list = np.array(imagine_list)
    stacked_list = np.vstack((real_list, imagine_list))
    return stacked_list, kernels, kernel_params


# In[12]:


train_selected_amount = 50000
test_selected_amount = 10000
num_classes = 10
gray_x_train = grayscale(x_train[:train_selected_amount])
gray_x_test = grayscale(x_test[:test_selected_amount])
y_train = np_utils.to_categorical(y_train[:train_selected_amount], num_classes)
y_test = np_utils.to_categorical(y_test[:test_selected_amount], num_classes)
print(y_train.shape)


# In[51]:


convolved_x_train, kernels, kernel_params = convolve(gray_x_train)
convolved_x_train = np.einsum('abcd->bcda', convolved_x_train)
convolved_x_test, kernels, kernel_params = convolve(gray_x_test)
convolved_x_test = np.einsum('abcd->bcda', convolved_x_test)
np.save('convolved_x_train_cv2', convolved_x_train)
np.save('convolved_x_test_cv2', convolved_x_test)
# print(convolved_x_train.shape)


# In[50]:


# selected = 7
# test = convolved_x_train[selected, :]
# print(test.shape)

# plt.figure()
# plt.subplot(121)
# plt.title('original image')
# plt.imshow(gray_x_train[selected], cmap='gray')

# # fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(15, 20))
# # plt.gray()
# # fig.suptitle('Gabor response images', fontsize=12)
# # idx = 0
# # for row in range(0, 8):
# #     for col in range(0, 6):
# #         print(kernels[idx].shape)
# #         cell = axes[row][col]
# #         cell.imshow(np.imag(kernels[idx]))
# #         idx += 1

# fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(15, 20))
# plt.gray()
# fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
# idx = 0
# for row in range(0, 8):
#     for col in range(0, 6):
#         cell = axes[row][col]
#         cell.imshow(test[::, ::, idx])
#         idx += 1

