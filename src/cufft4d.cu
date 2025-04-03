#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 64  // X Dimension
#define NY 64  // Y Dimension
#define NZ 512  // Z Dimension
#define NW 128  // W Dimension
#define TOTAL_ELEMENTS (NX * NY * NZ * NW)

// Function to execute 1D FFT along a specific dimension
void execute_fft(cufftComplex *d_data, int dim_size, int batch, int stride, int dist) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(&plan, 1, &dim_size, 
                              NULL, stride, dist, 
                              NULL, stride, dist, 
                              CUFFT_C2C, batch));

    // Perform FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftDestroy(plan));
}

int main() {
    cufftComplex *data, *d_data;
    size_t size = TOTAL_ELEMENTS * sizeof(cufftComplex);

    data = (cufftComplex *)malloc(size);
    for (unsigned int i = 0; i < TOTAL_ELEMENTS; i++) {
        data[i].x = rand() / (float)RAND_MAX;
        data[i].y = 0;
    }

    // Allocate device memory for 4D data
    CHECK_CUDA(cudaMalloc((void**)&d_data, size));

    CHECK_CUDA(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));

    // Perform FFT along each dimension sequentially
    execute_fft(d_data, NX, NY * NZ * NW, 1, NX);            // FFT along X
    execute_fft(d_data, NY, NX * NZ * NW, NX, NY);           // FFT along Y
    execute_fft(d_data, NZ, NX * NY * NW, NX * NY, NZ);      // FFT along Z
    execute_fft(d_data, NW, NX * NY * NZ, NX * NY * NZ, NW); // FFT along W

    // Free GPU memory
    CHECK_CUDA(cudaFree(d_data));

    printf("4D FFT execution completed successfully!\n");
    return 0;
}
