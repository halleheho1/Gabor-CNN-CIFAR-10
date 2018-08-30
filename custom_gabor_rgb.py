def custom_gabor(shape, dtype=None):
#    orientation_spread = np.linspace(0, 8, 8) / 4. * np.pi
    pi = np.pi
    orientation_spread = np.array([0, pi/4, pi/2, pi*3/4, pi, pi*5/4, pi*3/2, 2*pi])
    scales = np.linspace(2, 3, 6)
    real_kernels = []
#     size, sigma, theta, lambda, gamma aspect ratio
    for orientation in orientation_spread:
        for scale in scales:
            real_kernel = cv2.getGaborKernel((5, 5), 1, orientation, scale, 1, 0)
#             real_kernel = np.delete(np.delete(real_kernel, -1, 0), -1, 1)
            real_kernels.append(real_kernel)
    real_kernels = np.array([real_kernels, real_kernels, real_kernels])
    real_kernels = np.einsum('hijk->jkhi', real_kernels)
    print(real_kernels.shape)

    real_kernels = K.variable(real_kernels)
    random = K.random_normal(shape, dtype=dtype)
    return real_kernels
