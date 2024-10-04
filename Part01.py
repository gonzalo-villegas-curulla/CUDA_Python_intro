# %%

# Cuda kernel =def function that runs in a GPU device

#imports
from numba import cuda
import numpy as np

# A first kernel with decorator "cuda just in time"
@cuda.jit
def cudakernel0(array):
    for idx in range(array.size):
        array[idx] += 0.5 # Add this value to each 'dimension' of array

# Let us create a test array
array = np.array([0,1], dtype=np.float32)
print('The test array: ', array)

print('We launch the kernel using the test array\ncudakernel0[1,1](array)')
cudakernel0[1,1](array)
print('And the resulting uupdated kernel is: ', array)


# We were using the argukment [1,1] for the kernel. Those are "dimensions", more precisely,
# the first argument in brackets is the grid size
# the second argument is the blocksize

# kernelname[gridsize, blocksize](args)
# gridsize  = how many blocks constitute the grid 
# blocksize = how many threads in each block 
# [1,1] = 1 block x 1 thread/block = 1 thread

# %%
gridsize  = 1024
blocksize = 1024

print('Grid size: {}'.format(gridsize))
print('Block size: {}'.format(blocksize))