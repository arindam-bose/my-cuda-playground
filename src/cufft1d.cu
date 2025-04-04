#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define M_PI   3.14159265358979323846  /* pi */
#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_cufft_1d(unsigned int nx) {
    // Declaration
    float *samples;
    cufftComplex *complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan;

    size_t size = sizeof(cufftComplex) * nx;

    cudaEvent_t start, stop;
    float elapsed_time;

    // Allocate memory for the variables on the host
    samples = (float *)malloc(sizeof(float) * nx);
    complex_samples = (cufftComplex *)malloc(size);
    complex_freq = (cufftComplex *)malloc(size);

    // Input signal generation using cos(x)
    double delta = M_PI / 20.0;
    for (unsigned int i = 0; i < nx; i++) {
        samples[i] = cos(i * delta);
    }

    // Convert to a complex signal
    for (unsigned int i = 0; i < nx; i++) {
        complex_samples[i].x = samples[i];
        complex_samples[i].y = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Real data...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f\n", samples[i]);
        }
        printf("Complex data...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_samples[i].x, complex_samples[i].y);
        }
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

    // Setup the CUFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, nx, CUFFT_C2C, 1));
    
    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD));

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
    printf("%.6f\n", elapsed_time * 1e-3);

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    free(complex_freq);
    free(complex_samples);
    free(samples);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Error: This program requires exactly 1 command-line arguments.\n");
        return 1;
    }

    unsigned int nx = atoi(argv[1]);
    run_test_cufft_1d(nx);
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}