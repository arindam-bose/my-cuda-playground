#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>

#define N   1024  // dimension size
#define M_PI   3.14159265358979323846  /* pi */
#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_cufft_1d(int argc, char** argv) {
    // Declaration
    float *samples;
    cufftComplex *complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan;

    size_t size = sizeof(cufftComplex) * N;

    // Allocate memory for the variables on the host
    samples = (float *)malloc(sizeof(float) * N);
    complex_samples = (cufftComplex *)malloc(size);
    complex_freq = (cufftComplex *)malloc(size);

    // Input signal generation using cos(x)
    double delta = M_PI / 20.0;
    for (unsigned int i = 0; i < N; i++) {
        samples[i] = cos(i * delta);
    }

    // Convert to a complex signal
    for (unsigned int i = 0; i < N; i++) {
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

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_samples, size));
    CHECK_CUDA(cudaMalloc((void **)&d_complex_freq, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_samples, complex_samples, size, cudaMemcpyHostToDevice));

    // Setup the CUFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    
    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_freq, d_complex_freq, size, cudaMemcpyDeviceToHost));

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_freq[i].x, complex_freq[i].y);
        }
    }

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    free(complex_freq);
    free(complex_samples);
    free(samples);
}


int main(int argc, char **argv) {
    run_test_cufft_1d(argc, argv);
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}