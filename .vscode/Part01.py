# import sys
# import numba
# import numpy
from numba import cuda 
import numpy as np

@cuda.jit 
def cudakernel0(array):
    for idx in range(array.size):
        array[idx] += 0.5

array = np.array([0,1], np.float32)        
print('Initial array:', array)

print('Launch of kernel "cudakernel0[1,1](array)"')
cudakernel0[1,1](array)

print('Array after applying cuda kernel (updated)')
print('Array: ', array)

gridsize  = 1024
blocksize = 1024

cudakernel0[gridsize,blocksize](array) # All the threads try to execute the kernel on "the same array"
print('Array: ', array)

