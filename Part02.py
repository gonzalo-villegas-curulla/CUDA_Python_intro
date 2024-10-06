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

#  I just want to check something

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
