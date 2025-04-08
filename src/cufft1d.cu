#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define M_PI   3.14159265358979323846  /* pi */
#define PRINT_FLAG 0
#define NPRINTS 5  // print size

void printf_cufft_cmplx_array(cufftComplex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
}

float run_test_cufft_1d(unsigned int nx) {
    // Declaration
    float *samples;
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;
    cufftHandle plan;

    size_t size = sizeof(cufftComplex) * nx;

    cudaEvent_t start, stop;
    float elapsed_time;

    // Allocate memory for the variables on the host
    samples = (float *)malloc(sizeof(float) * nx);
    complex_data = (cufftComplex *)malloc(size);

    // Input signal generation using cos(x)
    double delta = M_PI / 20.0;
    for (unsigned int i = 0; i < nx; ++i) {
        samples[i] = cos(i * delta);
    }

    // Convert to a complex signal
    for (unsigned int i = 0; i < nx; ++i) {
        complex_data[i].x = samples[i];
        complex_data[i].y = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Real data...\n");
        for (unsigned int i = 0; i < NPRINTS; ++i) {
            printf("  %2.4f\n", samples[i]);
        }
        printf("Complex data...\n");
        printf_cufft_cmplx_array(complex_data, nx);
    }

    // Create CUDA events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_data, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // Setup the CUFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, nx, CUFFT_C2C, 1));
    
    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_data, d_complex_data, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_cufft_cmplx_array(complex_data, nx);
    }

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(complex_data);
    free(samples);

    return elapsed_time * 1e-3;
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Error: This program requires exactly 2 command-line arguments.\n");
        printf("       %s <arg0> <arg1>\n", argv[0]);
        printf("       arg0: FFT length in 1D\n");
        printf("       arg1: Number of iterations\n");
        printf("       e.g.: %s 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int niter = atoi(argv[2]);

    // Discard the first time running. It apparantly does some extra work during first time
    // JIT??
    run_test_cufft_1d(nx);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_cufft_1d(nx);
    }
    printf("%.6f\n", sum/(float)niter);
    
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}