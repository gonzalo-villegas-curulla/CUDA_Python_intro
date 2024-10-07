# %% WRITE a convolution kernel to filter an image

from numba import cuda
import numpy as np

@cuda.jit
def cuda_convo(RES, M, IM):
    # INPUTS:
    # 2D grid, 2D blocks (created inside kernel)
    # Mask M with odd numbers of rows and columns
    # A grayscale image IM

    # Thread coordinates:
    # cuda.grid(2) = (cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x ,
    #                 cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y )

    idx, jdx = cuda.grid(2)

    # For thread coords outside image, omit this thread
    im_rows, im_cols = IM.shape
    if (idx >= im_rows) or (jdx>= im_cols):
        return
    
    # To compute the convolution result at (idx, jdx) use:
    # (idx-delta_rows) ... (idx)th row ... (idx+delta_rows)
    # (jdx-delta_cols) ... (jdx)th col ... (jdx+delta_cols)
    # Floor division
    delta_rows = M.shape[0] // 2
    delta_cols = M.shape[1] // 2

    # S(idx,jdx) = \sum_{k,l} M[k,l] * IM[k-idx + d_row, jdx -l + d_col]
    #  and (k,l) span the mask array M
    S = 0
    for k in range(M.shape[0]):
        for l in range(M.shape[1]):
            idx_k = idx - k + delta_rows
            idx_l = jdx - l + delta_cols

            # Are we inside image?
            if (idx_k >= 0) and (idx_k < im_rows) and (idx_l < im_cols):
                S += M[k,l] * IM[idx_k, idx_l]
    #  Once we cover the whole mask span, store result at convolution location
    # print(idx,jdx)
    RES[idx,jdx] = S 

import skimage.data     
from skimage.color import rgb2gray 
# %matplotlib qt5
%matplotlib inline
import matplotlib.pyplot as plt

full_im = rgb2gray(skimage.data.coffee()).astype(np.float32) / 255
# plt.figure()
# plt.imshow(full_im, cmap='gray')
# plt.title('Full size image:')

im = full_im[150:350,200:400].copy()
# im = full_im
# plt.figure()
# plt.imshow(im, cmap='gray')
# plt.title('Part of the image we use:')
# plt.show()


## %% TRY the kernel on the IM
res   = np.empty_like(im)
mask  = np.random.rand(13,13).astype(np.float32)
mask /= mask.sum()
print('Mask dims:', mask.shape )
print('Mask first (3,3) elements:\n', mask[:3,:3])

# let's use a blocksize of 32x32 pixels
bsize = (32,32)
#  And the gridsize chose accordingly to cover the image surface
gsize = (im.shape[0] // bsize[0] + 1 ,
         im.shape[1] // bsize[1] + 1)
print(gsize, bsize)

cuda_convo[gsize, bsize](res, mask, im)

# Plot the convolved image
plt.figure()
plt.imshow(im, cmap='gray')
plt.figure()
plt.imshow(res, cmap='gray')

# %% Check the error
from scipy.ndimage.filters import convolve as scipy_convolve 
scipy_result = scipy_convolve(im, mask, mode='constant', cval=0.0, origin=0)
print('Max rel error:',np.max(np.abs(res-scipy_result)/np.abs(scipy_result)))
# plt.figure()
# plt.imshow(scipy_result,cmap='gray')

# %% TIMING our kernel
%timeit cuda_convo[gsize, bsize](res, mask, im)
scipy_result = np.empty_like(im)
%timeit scipy_convolve(im, mask, output=scipy_result, mode='constant', cval=0.0, origin=0)

# %% remove the memory transfer times
dev_im   = cuda.to_device(im)
dev_mask = cuda.to_device(mask)
dev_res  = cuda.to_device(res)

%timeit cuda_convo[gsize, bsize](dev_res, dev_mask, dev_im)
