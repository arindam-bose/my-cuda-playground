#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size

// Function to execute 1D FFT along a specific dimension
void execute_cufft1d(cufftComplex *d_idata, cufftComplex *d_odata, int dim_size, int batch, int stride, int dist) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(&plan, 1, &dim_size, 
                                NULL, stride, dist, 
                                NULL, stride, dist, 
                                CUFFT_C2C, batch));

    // Perform FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_idata, d_odata, CUFFT_FORWARD));
    CHECK_CUFFT(cufftDestroy(plan));
}

void run_test_cufft_4d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    cudaEvent_t start, stop;
    float elapsed_time;

    // Allocate memory for the variables on the host
    complex_samples = (cufftComplex *)malloc(size);
    complex_freq = (cufftComplex *)malloc(size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i].x = rand() / (float)RAND_MAX;
        complex_samples[i].y = 0;
    }

    // Create CUDA events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_samples, size));
    CHECK_CUDA(cudaMalloc((void **)&d_complex_freq, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_samples, complex_samples, size, cudaMemcpyHostToDevice));

    // Perform FFT along each dimension sequentially
    execute_cufft1d(d_complex_samples, d_complex_freq, nx, ny * nz * nw, 1, nx);         // FFT along X
    execute_cufft1d(d_complex_freq, d_complex_freq, ny, nx * nz * nw, nx, ny);           // FFT along Y
    execute_cufft1d(d_complex_freq, d_complex_freq, nz, nx * ny * nw, nx * ny, nz);      // FFT along Z
    execute_cufft1d(d_complex_freq, d_complex_freq, nw, nx * ny * nz, nx * ny * nz, nw); // FFT along W

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_freq, d_complex_freq, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_freq[i].x, complex_freq[i].y);
        }
    }

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Elapsed time: %.6f ms\n", elapsed_time);

    // Clean up
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(complex_freq);
    free(complex_samples);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 5 command-line arguments.\n");
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    int nw = atoi(argv[4]);
    run_test_cufft_4d(nx, ny, nz, nw);
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}