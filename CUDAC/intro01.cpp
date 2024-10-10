#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int idx = 0; idx < n; idx++)
        y[idx] = x[idx] + y[idx];
}

int main(void)
{
    int N = 1<<20; // 1M elements
    std::cout << N <<"\n";
    float *x = new float[N];
    float *y = new float[N];

    // Initialize the "x" and "y" arrays on the host
    for (int idx = 0; idx < N; idx++)
    {
        x[idx] = 1.0f;
        y[idx] = 2.0f;
        // std::cout<<idx<<std::endl;
    }

    // Run the kernel on 1M elements on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int idx=0; idx<N; idx++)
        maxError = fmax(maxError, fabs(y[idx] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;

}