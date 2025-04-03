#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size
#define IFFT_FLAG 0

void run_test_cufft_3d(unsigned int nx, unsigned int ny, unsigned int nz) {
    srand(2025);

    // Declaration
    cufftComplex *complex_samples, *new_complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan;

    unsigned int element_size = nx * ny * nz;
    size_t size = sizeof(cufftComplex) * element_size;

    // Allocate memory for the variables on the host
    complex_samples = (cufftComplex *)malloc(size);
    complex_freq = (cufftComplex *)malloc(size);
    if (IFFT_FLAG) {new_complex_samples = (cufftComplex *)malloc(size);}

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i].x = rand() / (float)RAND_MAX;
        complex_samples[i].y = 0;
    }

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_samples, size));
    CHECK_CUDA(cudaMalloc((void **)&d_complex_freq, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_samples, complex_samples, size, cudaMemcpyHostToDevice));

    // Setup a 3D FFT plan
    CHECK_CUFFT(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));

    // Execute the forward 3D FFT (in-place computation)
    CHECK_CUFFT(cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_freq, d_complex_freq, size, cudaMemcpyDeviceToHost));

    if (IFFT_FLAG) {
        // Execute the inverse 3D IFFT (in-place computation)
        CHECK_CUFFT(cufftExecC2C(plan, d_complex_freq, d_complex_samples, CUFFT_INVERSE));

        // Retrieve the results into host memory
        CHECK_CUDA(cudaMemcpy(new_complex_samples, d_complex_samples, size, cudaMemcpyDeviceToHost));

        // Normalize
        for (unsigned int i = 0; i < element_size; i++) {
            new_complex_samples[i].x /= (float)element_size;
            new_complex_samples[i].y /= (float)element_size;
        }
    }

    if (PRINT_FLAG && IFFT_FLAG) {
        printf("Complex samples after FFT and IFFT...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f -> %2.4f + i%2.4f\n", complex_samples[i].x, complex_samples[i].y, new_complex_samples[i].x, new_complex_samples[i].y);
        }
    }

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    if (IFFT_FLAG) {free(new_complex_samples);}
    free(complex_freq);
    free(complex_samples);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Error: This program requires exactly 3 command-line arguments.\n");
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    run_test_cufft_3d(nx, ny, nz);
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}