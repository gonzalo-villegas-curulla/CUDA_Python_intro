#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:13:46 2024

@author: GVC
"""

from numba import cuda
import numpy as np
import warnings
warnings.filterwarnings("ignore")



@cuda.jit
def cudakernel0(array):
    for i in range(array.size):
        array[i] += 0.5

if 0:        
    array = np.array([0,1], np.float32)
    print('Initial array: ', array)
    
    print('Kernel launch: cudakernel0[1, 1](array)')
    # for i in range(int(1e4)):
    cudakernel0[1, 1](array)
    
    print('Updated array:', array)


# =====

# kernelname[ gridsize, blocksize](argument)
# kernelname[ NumOfBlockInGrid, NumOfThreadsPerBlock](argument)
# TOTALthreads = gridsize x blocksize

if 0:
    array = np.array([0, 1], np.float32)
    print('initial array:', array)
    
    gridsize = 1024
    blocksize = 1024
    print("Grid size: {}, Blocksize: {}".format(gridsize,blocksize))
    
    print("Total number of threads:", gridsize*blocksize)
    
    print("Kernel launch: cudakernel0[gridsize,blocksize](array)")
    cudakernel0[gridsize,blocksize](array)
    
    print("Updated array:", array)


@cuda.jit 
def cudakernel1(array):
    thread_position = cuda.grid(1)
    array[thread_position] += 0.5
    

# Array os size 2, use 2 threads
array = np.array([0, 1], np.float32)
gsize = 1
bsize = 2
cudakernel1[gsize, bsize](array)    
# print('updated array:', array)

# Array of size 2^20, needing 2^20 threads (1024*1024)
array = np.zeros(1024*1024, np.float32)
gsize = 1024
bsize = 1024
print('Initial array:', array)
cudakernel1[gsize, bsize](array)

print('Updated array:', array)

# ===================================
"""
grid \in 3D

block has dimensions:
    (cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z)

grid has dimensions:
    (cuda.gridDim.x, cuda.gridDim.y, cuda.gridDim.z)

NumThrPerBlock = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z

Grid has:

NumBlocks = cuda.gridDim.x * cuda.gridDim.y * cuda.gridDim.z

kernel_name[(griddimx,griddimy,griddimz), (blockdimx,blockdimy,blockdimz)](arguments)    


cudakernel1[1024, 1024](array)
<==>
cudakernel1[(1024,1,1),(1024,1,1)](array)


__CUDA__

POSITIONS of the current thread inside the current block
cuda.threadIdx.x
cuda.threadIdx.y
cuda.threadIdx.z

POSITIONS of the current block inside the grid
cuda.blockIdx.x
cuda.blockIdx.y
cuda.blockIdx.z


Absolute POSITION of a thread inside the grid:
    
    (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x,
     cuda.blockIdx.y * cuda.BlockDim.y + cuda.threadIdx.y, 
     cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z)
    
    which is returned by
    
    cuda.grid(3)
"""


"""
cuda.grid(2) returns the tuple

    (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x, 
     cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y)
    
and cuda.grid(1) :

    (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)    
"""


# ==== TIMING PERFORMANCE ========

@cuda.jit 
def cudakernel1(array):
    thread_position = cuda.grid(1)
    array[thread_position] += 0.5
    
@cuda.jit()
def cudakernel1b(array):
    thread_position = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x 
    array[thread_position]     += 0.5
    
if 0:    
    a = np.random.randn(2048*2048).astype(np.float32)
    p = np.float32(range(1,10))    
    c = p[::-1]
    
    # %timeit np.polyval(p, a)
    # %timeit np.polynomial.polynomial.polyval(a, c)
        
    print('Maximum absolute difference:', np.max(np.abs(np.polyval(p,a)-np.polynomial.polynomial.polyval(a,c))))
    
    # %timeit np.polynomial.polynomial.polyval(a,c,tensor=False)


# Now, use a cuda kernel

# @cuda.jit 
# def cuda_polyval(result, array, coeffs):
#     # Coeffs in descending order
#     idx = cuda.grid(1)
#     val = coeffs[0]
#     for coeff in coeffs[1:]:
#         val = val * array[idx] + coeff
#     result[idx] = val
    
@cuda.jit
def cuda_polyval(result, array, coeffs):
    # Evaluate a polynomial function over an array with Horner's method.
    # The coefficients are given in descending order.
    i = cuda.grid(1) # equivalent to i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    val = coeffs[0]
    for coeff in coeffs[1:]:
        val = val * array[i] + coeff
    result[i] = val    


array = np.random.randn(2048*1024).astype(np.float32)
# array = np.random.randn(2**10).astype(np.float32)
coeffs = np.float32(range(1,10))
result = np.empty_like(array)

cuda_polyval[2048,1024](result, array, coeffs)
numpy_result = np.polyval(coeffs, array)

print('Max rel error compared to numpy.polyval:', np.max(np.abs(numpy_result-result)))

# %timeit cuda_polyval[2048,1024](result, array, coeffs)


# Time the the time on the GPU only:
    
d_array = cuda.to_device(array)
d_coeffs = cuda.to_device(coeffs)
d_result = cuda.to_device(result)

# %timeit cuda_polyval[2048,1024](d_result, d_array, d_coeffs)
# %timeit cuda_polyval[2048, 1024](d_result, d_array, d_coeffs)
