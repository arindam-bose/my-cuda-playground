#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 64  // X-dimension size
#define NY 64  // Y-dimension size
#define NZ 64  // Z-dimension size

int main() {
    cufftHandle plan;
    cufftComplex *d_data;
    size_t size = NX * NY * NZ * sizeof(cufftComplex);

    // Allocate device memory for 3D data
    cudaMalloc((void**)&d_data, size);

    // Create a 3D FFT plan
    if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C) != CUFFT_SUCCESS) {
        printf("CUFFT plan creation failed!\n");
        return -1;
    }

    // Execute the forward FFT (in-place computation)
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        printf("CUFFT forward execution failed!\n");
        return -1;
    }

    // Execute the inverse FFT (to recover original data)
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        printf("CUFFT inverse execution failed!\n");
        return -1;
    }

    // Normalize the output (since inverse FFT scales the result by NX*NY*NZ)
    int total_elements = NX * NY * NZ;
    cudaDeviceSynchronize();

    printf("3D FFT execution completed successfully!\n");

    // Clean up
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}