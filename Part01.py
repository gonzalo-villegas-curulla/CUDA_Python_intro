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
array = np.array([0.,1], dtype=np.float32)

print('Initial array: ', array)
print('Grid size: {}'.format(gridsize))
print('Block size: {}'.format(blocksize))
print('Total number of threads: {} (gsize x bsize)'.format(gridsize*blocksize))
print('Run kernel with  gsize and bsize: kudakernel0[gsize,bsize](array)')
cudakernel0[gridsize, blocksize](array)
print('Updated array: ', array)

# The result changes every time we run the kernel! Out of the total gsize*bsize threads,
# two threads may read the array at the same time ==> same result
# but one thread may write the output before another thread reads the array ==> different result

# %% Now let's start indexing 

@cuda.jit 
def cudakernel1(array):
    thread_pos = cuda.grid(1) # the "1" is a dimensional indexing request
    # This gives a position pointer within the grid
    array[thread_pos] += 0.5 

array = np.array([0,1], dtype=np.float32)    
print('Initial array: ', array)
print('Launch the kernel1[1,2]()\n 1 block, 2 threads per block')
cudakernel1[1,2](array)
print('Array updated: ', array)

print('Launching cudakernel1[1,1]() will only update\n the first cell of the array')
cudakernel1[1,1](array)
print('Updated array: ', array,' see?')



# %% Now leverage from the fact that 2^20 = 1024*1024

gsize = 1024
bsize = 1024

# Let's add 0.5 to a 1024x1024 array of zeros
array = np.zeros(gsize*bsize, dtype=np.float32)
print('The array zeros(1024x1024) at init:', array)

print('Launch kernel cudakernel1[1024,1024](array)')
cudakernel1[1024,1024](array)
print('Updated array: ', array)

# Let's compare them
print('The result: ', np.all(array == 0.5+np.zeros(gsize*bsize, dtype=np.float32) ) )

