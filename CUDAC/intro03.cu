#include <iostream>
#include <math.h>
#include <cstdio>
#include <cuda_runtime.h>

__global__
void add(int n, float *x, float *y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<n){
        y[idx] = x[idx] + y[idx];
    }
}


int main(void){

    int N = 1<<30;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // inits
    float val1 = 1.0, val2 = 2.0;
    for (int idx=0; idx<N; idx++){
        x[idx] = val1;
        y[idx] = val2;
    }

    int blocksize =1024;
    int gridsize = (N + blocksize - 1)/blocksize;

    std::cout << "Gsize: " << gridsize << " blocks. Bsize: " << blocksize << " threads per block." << std::endl; 
    add<<<gridsize, blocksize>>>(N, x, y);

    //handle kernel cuda errors

    //any error after synching CPU and GPU?
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        std::cerr << "cuda error after synch: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // errorsGeorge in the kernel launch?
    err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr << "cuda kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }


    // value error check
    float maxErr = 0.0f;
    float total = val1+val2;
    for (int idx=0; idx<N; idx++){
        maxErr = fmax(maxErr, fabs(y[idx]- total));
    }
    std::cout << "MaxError: " << maxErr << std::endl;

    return 0;
}
