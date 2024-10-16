#include <iostream>
#include <math.h>
#include <cstdio>
#include <cuda_runtime.h>

// function for the GPU device 
__global__
void add(int n, float *x, float *y, float *z){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;  
   if (idx<n){
        z[idx] = 9.0;
   }
}

int main(void)
{
    int N = 10; // 1<<20; // 1M elements
    //std::cout << N <<"\n";

    //  Memory allocations:
    // we use unified memory (accessible by cpu and gpu) and call cudaMallocManaged() returning a pointer
    float *x, *y, *z;

    // allocate at addresses &x and &y, size N, type float
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&z, N*sizeof(float));


    // Initialize the "x" and "y" arrays on the host and populate them
    for (int idx = 0; idx < N; idx++)
    {
        x[idx] = 1.0f;
        y[idx] = 2.0f;
        z[idx] = 12.0f;
        printf("Before kernel, z[%d]=%f\n", idx, z[idx]);
        // std::cout<<idx<<std::endl;
    }

    // Run the kernel on 1M elements on the CPU/GPU
    //  NOW need to specify gridsize and blocksize inside <<<,>>>

    //add<<<1,1>>>(N, x, y,z); // which is using a grid of one block and one thread per block

    int bsize = 10;
    int gsize = (N+bsize-1)/bsize;
    add<<<gsize,bsize>>>(N,x,y,z);

    cudaError_t err = cudaDeviceSynchronize(); //  synchronize  


    std::cout << err << std::endl;

    std::cout << std::endl;
    std::cout << "Gsize: "<<gsize << " block(s). Bsize: " << bsize << " threads per block."<< std::endl;

    //  Before continuining, we need to synchronise CPU and GPU. The kernel launch does not prevent
    // processes in CPU from stopping. With:




    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int idx=0; idx<N; idx++) {
        maxError = fmax(maxError, fabs(z[idx] - 3.0f));
        std::cout << z[idx] << std::endl;
}
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;

}
