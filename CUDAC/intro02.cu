#include <iostream>
#include <math.h>

// function for the GPU device 
__global__
void add(int n, float *x, float *y)
{
    for (int idx = 0; idx < n; idx++)
        y[idx] = x[idx] + y[idx];
}

int main(void)
{
    int N = 1<<20; // 1M elements
    std::cout << N <<"caca\n";
    // float *x = new float[N];
    // float *y = new float[N];

    //  Memory allocations:
    // we use unified memory (accessible by cpu and gpu) and call cudaMallocManaged() returning a pointer
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));




    // Initialize the "x" and "y" arrays on the host
    for (int idx = 0; idx < N; idx++)
    {
        x[idx] = 1.0f;
        y[idx] = 2.0f;
        // std::cout<<idx<<std::endl;
    }

    // Run the kernel on 1M elements on the CPU
    // add(N, x, y);
    //  NOW need to specify gridsize and blocksize inside <<,>>
    add<<<1,1>>>(N, x, y); // which is using a grid of one block and one thread per block

    //  Before continuining, we need to synchronise CPU and GPU. The kernel launch does not prevent
    // processes in CPU from stopping. With:

    cudaDeviceSynchronize(); // the CPU will wait for the kernel to finish 



    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int idx=0; idx<N; idx++)
        maxError = fmax(maxError, fabs(y[idx] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;

}