#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import skimage.data
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve as scipy_convolve


@cuda.jit
def convolve(result, mask, image):

    idx, jdx = cuda.grid(2)

    image_rows, image_cols = image.shape
    if (idx >= image_rows) or (jdx >= image_cols):
        return
    delta_rows = mask.shape[0] // 2
    delta_cols = mask.shape[1] // 2

    S = np.float32(0)
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            idx_k = idx - k + delta_rows
            jdx_l = jdx - l + delta_cols
            # (-4-) Check if (i_k, j_k) coordinates are inside the image:
            if (idx_k >= 0) and (idx_k < image_rows) and (jdx_l >= 0) and (jdx_l < image_cols):
                S += mask[k, l] * image[idx_k, jdx_l]
    result[idx, jdx] = S


if __name__ == '__main__':
    # Read
    full_image = rgb2gray(skimage.data.coffee()).astype(np.float32) / 255
    image = full_image[150:350, 200:400].copy()

    # Preallocate
    result = np.empty_like(image)

    # Mask:
    mask = np.random.rand(13, 13).astype(np.float32)
    mask /= mask.sum()  # Normalize

    # We use blocks of 32x32 pixels:
    bsize = (32, 32)

    # We compute grid dimensions big enough to cover the whole image:
    gsize = (image.shape[0] // bsize[0] + 1, image.shape[1] // bsize[1] + 1)

    # We apply our convolution to our image:
    convolve[gsize, bsize](result, mask, image)

    # We check that the error with respect to Scipy convolve function is small:
    scipy_result = scipy_convolve(image, mask, mode='constant', cval=0.0, origin=0)
    max_rel_error = np.max(np.abs(result - scipy_result) / np.abs(scipy_result))
    if  max_rel_error > 1e-5:
        raise AssertionError('Maximum relative error w.r.t Scipy convolve is too large: ' 
                             + max_rel_error)