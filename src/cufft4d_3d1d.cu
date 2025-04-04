#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_cufft_4d_alt(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);
    
    // Declaration
    cufftComplex *complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan3d, plan1d;

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

    // Print input stuff
    if (PRINT_FLAG) {
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

    // -----------------------
    // 1. Perform 3D FFTs over each W slice (W batches of 3D volumes)
    // -----------------------
    CHECK_CUFFT(cufftPlan3d(&plan3d, nx, ny, nz, CUFFT_C2C));
    for (int w = 0; w < nw; ++w) {
        size_t offset = w * nx * ny * nz;
        CHECK_CUFFT(cufftExecC2C(plan3d, d_complex_samples + offset, d_complex_freq + offset, CUFFT_FORWARD));
    }

    // -----------------------
    // 2. Perform 1D FFT along W dimension
    // -----------------------
    // There are NX*NY*NZ such transforms (one for each (x,y,z) point)
    int n[1] = { (int)nw };
    int batch = nx * ny * nz;
    int stride = 1;
    int dist = nw;

    CHECK_CUFFT(cufftPlanMany(&plan1d, 1, n,       // rank, dimensions
                                NULL, stride, dist,
                                NULL, stride, dist,
                                CUFFT_C2C, batch));

    // Execute the 1D FFTs (in-place)
    CHECK_CUFFT(cufftExecC2C(plan1d, d_complex_freq, d_complex_freq, CUFFT_FORWARD));

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
    printf("Elapsed time: %.6f s\n", elapsed_time * 1e-3);

    // Cleanup
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan3d));
    CHECK_CUFFT(cufftDestroy(plan1d));
    free(complex_freq);
    free(complex_samples);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 4 command-line arguments.\n");
        return 1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int nz = atoi(argv[3]);
    unsigned int nw = atoi(argv[4]);
    run_test_cufft_4d_alt(nx, ny, nz, nw);
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}