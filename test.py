import cupy as cp
import time

# Create two large random matrices
a = cp.random.rand(1000, 1000)
b = cp.random.rand(1000, 1000)

# Perform matrix multiplication	
start = time.time()

while 1 < 10:
	c = cp.dot(a, b)
end = time.time()

print(f"Matrix multiplication completed in {end - start} seconds")
