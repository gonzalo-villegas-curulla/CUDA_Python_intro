#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import skimage.data
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve as scipy_convolve


@cuda.jit
def convolve(result, mask, image):
    # expects a 2D grid and 2D blocks,
    # a mask with odd numbers of rows and columns, (-1-)
    # a grayscale image

    # (-2-) 2D coordinates of the current thread:
    i, j = cuda.grid(2)

    # (-3-) if the thread coordinates are outside of the image, we ignore the thread:
    image_rows, image_cols = image.shape
    if (i >= image_rows) or (j >= image_cols):
        return

    # To compute the result at coordinates (i, j), we need to use delta_rows rows of the image
    # before and after the i_th row,
    # as well as delta_cols columns of the image before and after the j_th column:
    delta_rows = mask.shape[0] // 2
    delta_cols = mask.shape[1] // 2

    # The result at coordinates (i, j) is equal to
    # sum_{k, l} mask[k, l] * image[i - k + delta_rows, j - l + delta_cols]
    # with k and l going through the whole mask array:
    s = 0
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            # (-4-) Check if (i_k, j_k) coordinates are inside the image:
            if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):
                s += mask[k, l] * image[i_k, j_l]
    result[i, j] = s


if __name__ == '__main__':
    # Read image
    full_image = rgb2gray(skimage.data.coffee()).astype(np.float32) / 255
    image = full_image[150:350, 200:400].copy()

    # We preallocate the result array:
    result = np.empty_like(image)

    # We choose a random mask:
    mask = np.random.rand(13, 13).astype(np.float32)
    mask /= mask.sum()  # We normalize the mask

    # We use blocks of 32x32 pixels:
    blockdim = (32, 32)

    # We compute grid dimensions big enough to cover the whole image:
    griddim = (image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] + 1)

    # We apply our convolution to our image:
    convolve[griddim, blockdim](result, mask, image)

    # We check that the error with respect to Scipy convolve function is small:
    scipy_result = scipy_convolve(image, mask, mode='constant', cval=0.0, origin=0)
    max_rel_error = np.max(np.abs(result - scipy_result) / np.abs(scipy_result))
    if  max_rel_error > 1e-5:
        raise AssertionError('Maximum relative error w.r.t Scipy convolve is too large: ' 
                             + max_rel_error)