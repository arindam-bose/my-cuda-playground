#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 16  // Dimension X
#define NY 16  // Dimension Y
#define NZ 16  // Dimension Z
#define NW 16  // Dimension W
#define TOTAL_ELEMENTS (NX * NY * NZ * NW)

// Error handling macro
#define CUDA_CALL(call) \
    if((call) != cudaSuccess) { \
        printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
        return -1; \
    }

#define CUFFT_CALL(call) \
    if((call) != CUFFT_SUCCESS) { \
        printf("CUFFT error at %s:%d\n", __FILE__, __LINE__); \
        return -1; \
    }

int main() {
    cufftHandle plan;
    cufftComplex *d_data;
    size_t size = TOTAL_ELEMENTS * sizeof(cufftComplex);
    int NN[4] = {NX, NY, NZ, NW};

    // Allocate memory on GPU
    CUDA_CALL(cudaMalloc((void**)&d_data, size));

    // Create FFT plan for the first dimension (X)
    CUFFT_CALL(cufftPlanMany(&plan, 1, NN, 
                             NULL, 1, NX, // Input batch size
                             NULL, 1, NX, // Output batch size
                             CUFFT_C2C, NY * NZ * NW));

    // Execute FFT along X
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan));

    // Create FFT plan for the second dimension (Y)
    CUFFT_CALL(cufftPlanMany(&plan, 1, NN, 
                             NULL, NX, NY, 
                             NULL, NX, NY, 
                             CUFFT_C2C, NX * NZ * NW));

    // Execute FFT along Y
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan));

    // Create FFT plan for the third dimension (Z)
    CUFFT_CALL(cufftPlanMany(&plan, 1, NN, 
                             NULL, NX * NY, NZ, 
                             NULL, NX * NY, NZ, 
                             CUFFT_C2C, NX * NY * NW));

    // Execute FFT along Z
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan));

    // Create FFT plan for the fourth dimension (W)
    CUFFT_CALL(cufftPlanMany(&plan, 1, NN, 
                             NULL, NX * NY * NZ, NW, 
                             NULL, NX * NY * NZ, NW, 
                             CUFFT_C2C, NX * NY * NZ));

    // Execute FFT along W
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan));

    // Free GPU memory
    CUDA_CALL(cudaFree(d_data));

    printf("4D FFT execution completed successfully!\n");

    return 0;
}
