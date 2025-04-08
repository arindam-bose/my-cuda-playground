#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 2  // Dimension X
#define NY 2  // Dimension Y
#define NZ 2  // Dimension Z
#define NW 2  // Dimension W

#define TOTAL_SIZE (NX * NY * NZ * NW)

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
        return EXIT_FAILURE; \
    }

#define CHECK_CUFFT(call) \
    if ((call) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d\n", __FILE__, __LINE__); \
        return EXIT_FAILURE; \
    }

int main() {
    srand(2025);
    cufftComplex *h_data, *d_data;
    cufftHandle plan;
    
    h_data = (cufftComplex *)malloc(sizeof(cufftComplex) * TOTAL_SIZE);
    for (unsigned int i = 0; i < TOTAL_SIZE; ++i) {
        h_data[i].x = rand() / (float)RAND_MAX;
        h_data[i].y = 0;
    }
    printf("Input...\n");
    for (unsigned int i = 0; i < 16; ++i) {
        printf("  %2.4f + i%2.4f\n", h_data[i].x, h_data[i].y);
    }

    // Allocate memory
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(cufftComplex) * TOTAL_SIZE));

    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Initialize plan dimensions
    int rank = 1;       // We're doing 1D FFTs
    int n[1];           // FFT length
    int istride = 1;    
    int idist;
    int inembed[1];

    // ------------ FFT along W --------------
    n[0] = NW;
    inembed[0] = NW;
    idist = 1;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n, 
                              inembed, istride, idist,
                              inembed, istride, idist,
                              CUFFT_C2C, NX * NY * NZ));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);

    // ------------ FFT along Z --------------
    n[0] = NZ;
    inembed[0] = NZ * NW;
    idist = NW;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n,
                              inembed, istride, idist,
                              inembed, istride, idist,
                              CUFFT_C2C, NX * NY));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);

    // ------------ FFT along Y --------------
    n[0] = NY;
    inembed[0] = NY * NZ * NW;
    idist = NZ * NW;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n,
                              inembed, istride, idist,
                              inembed, istride, idist,
                              CUFFT_C2C, NX));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);

    // ------------ FFT along X --------------
    n[0] = NX;
    inembed[0] = NX * NY * NZ * NW;
    idist = NY * NZ * NW;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n,
                              inembed, istride, idist,
                              inembed, istride, idist,
                              CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cufftDestroy(plan);

    CHECK_CUDA(cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    printf("Output...\n");
    for (unsigned int i = 0; i < 16; ++i) {
        printf("  %2.4f + i%2.4f\n", h_data[i].x, h_data[i].y);
    }

    CHECK_CUDA(cudaFree(d_data));

    printf("4D FFT complete using chained 1D FFTs.\n");
    return 0;
}
