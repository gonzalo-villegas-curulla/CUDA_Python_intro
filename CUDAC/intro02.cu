#include <iostream>
#include <math.h>
#include <cstdio>
#include <cuda_runtime.h>

// function for the GPU device 
__global__
void add(int n, float *x, float *y){
   //int idx = blockIdx.x * blockDim.x + threadIdx.x;  
   //if (idx<n){
   //     y[idx] = x[idx] + y[idx];
   //}
    printf("Hola\n");
   for (int idx=0; idx<n; idx++){
        y[idx] = x[idx] + y[idx];
   }
}

int main(void)
{
    int N =  1<<16; // ($1)*2^($2) elements
    //std::cout << N <<"\n";

    //  Memory allocations:
    // we call cudaMallocManaged() to use UNIFIED memory accessible for CPU and GPU
    float *x, *y;

    // allocating at addresses &x and &y, size N, type float
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));


    // Initialize the "x" and "y" arrays on the host and populate them
    for (int idx = 0; idx < N; idx++)
    {
        x[idx] = 1.0f;
        y[idx] = 2.0f;
    }

    // Run the kernel on 1M elements on the CPU/GPU
    //  NOW need to specify gridsize and blocksize inside <<<,>>>
    int bsize = 256;
    int gsize = (N+bsize-1)/bsize;
    add<<<gsize,bsize>>>(N,x,y);

    // CUDA error message check after kernel launch
    cudaError_t err = cudaDeviceSynchronize(); //  synchronize  
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after synch: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // kernel launch errors?
    err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr << "CUDA kernel launch err: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Gsize: "<<gsize << " block(s). Bsize: " << bsize << " threads per block."<< std::endl;


    // value error check 
    float maxError = 0.0f;
    for (int idx=0; idx<N; idx++) {
        maxError = fmax(maxError, fabs(y[idx] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;

}
