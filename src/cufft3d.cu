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
    cufftComplex *complex_samples;
    cufftComplex *new_complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan;

    unsigned int element_size = nx * ny * nz;

    // Allocate memory for the variables on the host
    complex_samples = (cufftComplex *)malloc(sizeof(cufftComplex) * element_size);
    complex_freq = (cufftComplex *)malloc(sizeof(cufftComplex) * element_size);
    if (IFFT_FLAG) {new_complex_samples = (cufftComplex *)malloc(sizeof(cufftComplex) * element_size);}

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i].x = rand() / (float)RAND_MAX;
        complex_samples[i].y = 0;
    }

    // Allocate device memory for complex signal and output frequency
    cudaMalloc((void **)&d_complex_samples, sizeof(cufftComplex) * element_size);
    cudaMalloc((void **)&d_complex_freq, sizeof(cufftComplex) * element_size);

    // Copy host memory to device
    cudaMemcpy(d_complex_samples, complex_samples, sizeof(cufftComplex) * element_size, cudaMemcpyHostToDevice);

    // Setup a 3D FFT plan
    cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C);

    // Execute the forward 3D FFT (in-place computation)
    cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD);

    // Retrieve the results into host memory
    cudaMemcpy(complex_freq, d_complex_freq, sizeof(cufftComplex) * element_size, cudaMemcpyDeviceToHost);

    if (IFFT_FLAG) {
        // Execute the inverse 3D IFFT (in-place computation)
        cufftExecC2C(plan, d_complex_freq, d_complex_samples, CUFFT_INVERSE);

        // Retrieve the results into host memory
        cudaMemcpy(new_complex_samples, d_complex_samples, sizeof(cufftComplex) * element_size, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

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
    cufftDestroy(plan);
    cudaFree(d_complex_freq);
    cudaFree(d_complex_samples);
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
    return 0;
}