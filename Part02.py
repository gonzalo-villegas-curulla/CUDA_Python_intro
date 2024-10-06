# %%
# Imports
from numba import cuda
import numpy as np

# define a kernel

arr = np.random.randn(2**11 * 2**10).astype(np.float32)

#  Polynomial coefficients:
p = np.float32(range(1,10)) # poly coefficients in descending order
c = p[::-1] # poly coeffs in ascending order

print(p)
print(c)

# P(x) = 0x^9 + 1x ^8 + 2x ^7 + 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x^2 + 9x^0


%timeit np.polyval(p, arr)
%timeit np.polynomial.polynomial.polyval(arr, c)

print('Max abs diff:', np.max( np.abs(np.polyval(p,arr)- np.polynomial.polynomial.polyval(arr,c))))



%timeit np.polynomial.polynomial.polyval(arr, c)
%timeit np.polynomial.polynomial.polyval(arr, c, tensor=True)



# %%
#  I just want to check that [10,10]=[(10,1,1),(10,1,1)]

@cuda.jit
def mykernel(arr):
    ptr = cuda.grid(1)
    arr[ptr] += 0.5

arr = np.array([0,1], dtype=np.float32)    
print(arr)
mykernel[10,10](arr)
print(arr)
mykernel[(10,1,1),(10,1,1)](arr)
print(arr)



# %%  NOW LET'S SEE IF WE CAN BEAT IT WITH A KERNEL OF OUR OWN

@cuda.jit
def cuda_polyval(res, arr, coeffs):
    idx_grid1 = cuda.grid(1)
    # cuda.grid(1) = cuda.blockIdx.x * cuda.gridDim.x + cuda.threadIdx.x
    V = coeffs[0] # Initialize the sumation-accumulator
    for C in coeffs[1:]: # exclude the first one, 0th
        V = C + arr[idx_grid1]*V # Look at the update of Horner's rule for polynomial evaluation
    res[idx_grid1] = V # Dump it into result

myarr  = np.random.randn(2**11 * 2**10).astype(np.float32)    
mycoeffs = np.float32(range(1,10))
result = np.empty_like(myarr)

gsize = 2**11
bsize = 2**10 
cuda_polyval[gsize, bsize](result, myarr, mycoeffs)
numpy_res = np.polyval(mycoeffs, myarr)

print('Max rel error compared to np.polyval: ', np.max(np.abs(numpy_res-result)/np.abs(numpy_res)))
%timeit cuda_polyval[gsize,bsize](result,myarr,mycoeffs)


# %% NOW WE REMOVE THE TRANSFER TIME from host memory to device memory and back

# transfer arrays to allocated memory in the device with cuda.to_device(object)
dev_array  = cuda.to_device(myarr)
dev_coeffs = cuda.to_device(mycoeffs)
dev_result = cuda.to_device(result)
# and use those to evaluate the kernel

%timeit cuda_polyval[gsize, bsize](dev_result, dev_array, dev_coeffs)

# %% Time the whole process now
%timeit dev_array  = cuda.to_device(myarr); dev_coeffs = cuda.to_device(mycoeffs); dev_result = cuda.to_device(result); cuda_polyval[gsize, bsize](dev_result, dev_array, dev_coeffs)                            

# %% Let's compare it with numba and preallocated memory

from numba import jit

@jit
def host_polyval(result, array, coeffs):
    for idx in range(len(array)):
        val = coeffs[0]
        for coeff in coeffs[1:]:
            val = coeff + array[idx]*val
        result[idx] = val

host_polyval(result, myarr, mycoeffs)        
print('Max abs diff with numpy.polyval', np.max(np.abs(np.polyval(mycoeffs,myarr) - result)))

%timeit host_polyval(result, myarr, mycoeffs)