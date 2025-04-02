#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 64  // X-dimension size
#define NY 64  // Y-dimension size
#define NZ 64  // Z-dimension size
#define ELEM_SIZE NX * NY * NZ

void run_test_cufft_3d(int argc, char** argv) {
    // Declaration
    cufftComplex *complex_samples, *new_complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;
    cufftHandle plan;
    srand(2025);

    // Allocate memory for the variables on the host
    complex_samples = (cufftComplex *)malloc(sizeof(cufftComplex) * ELEM_SIZE);
    complex_freq = (cufftComplex *)malloc(sizeof(cufftComplex) * ELEM_SIZE);
    new_complex_samples = (cufftComplex *)malloc(sizeof(cufftComplex) * ELEM_SIZE);

    for (unsigned int i = 0; i < ELEM_SIZE; i++) {
        complex_samples[i].x = rand() / (float)RAND_MAX;
        complex_samples[i].y = 0;
    }

    // Allocate device memory for complex signal and output frequency
    cudaMalloc((void **)&d_complex_samples, sizeof(cufftComplex) * ELEM_SIZE);
    cudaMalloc((void **)&d_complex_freq, sizeof(cufftComplex) * ELEM_SIZE);

    // Copy host memory to device
    cudaMemcpy(d_complex_samples, complex_samples, sizeof(cufftComplex) * ELEM_SIZE, cudaMemcpyHostToDevice);

    // Setup a 3D FFT plan
    cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C);

    // Execute the forward 3D FFT (in-place computation)
    cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD);

    // Retrieve the results into host memory
    cudaMemcpy(complex_freq, d_complex_freq, sizeof(cufftComplex) * ELEM_SIZE, cudaMemcpyDeviceToHost);

    // Execute the inverse 3D FFT (in-place computation)
    cufftExecC2C(plan, d_complex_freq, d_complex_samples, CUFFT_INVERSE);

    // Retrieve the results into host memory
    cudaMemcpy(new_complex_samples, d_complex_samples, sizeof(cufftComplex) * ELEM_SIZE, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < ELEM_SIZE; i++) {
        new_complex_samples[i].x /= (float)ELEM_SIZE;
        new_complex_samples[i].y /= (float)ELEM_SIZE;
    }

    for (unsigned int i = 0; i < 30; i++) {
        printf("  %2.4f + i%2.4f -> %2.4f + i%2.4f\n", complex_samples[i].x, complex_samples[i].y, new_complex_samples[i].x, new_complex_samples[i].y);
    }

    // Clean up
    cufftDestroy(plan);
    cudaFree(d_complex_freq);
    cudaFree(d_complex_samples);
    free(new_complex_samples);
    free(complex_freq);
    free(complex_samples);
}


int main(int argc, char **argv) {
    run_test_cufft_3d(argc, argv);
    return 0;
}