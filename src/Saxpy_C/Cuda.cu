#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define BLOCK_SIZE 1024
#define N 536870912

__global__ void saxpy(float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        y[i] = a * x[i] + y[i];
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(void)
{
    float *x, *y;
    float a = 2.0f;
    size_t size = N * sizeof(float);

    // Allocate memory for input and output vectors
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on the GPU and measure time

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    double start_time = get_time();
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(a, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    double end_time = get_time();

    double elapsed= end_time - start_time;

    // Print results and timing information
    printf("Elapsed time: %.6f seconds\n", elapsed);
    for (int i = 0; i < 10; ++i)
        printf("y[%d] = %f\n", i, y[i]);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
