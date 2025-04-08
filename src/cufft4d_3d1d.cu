#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define PRINT_FLAG 1
#define NPRINTS 16  // print size

void printf_cufft_cmplx_array(cufftComplex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i].x, complex_array[i].y);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i].x, complex_array[i].y);
    }
}

float run_test_cufft_4d_3d1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);
    
    // Declaration
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;
    cufftHandle plan3d, plan1d;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    cudaEvent_t start, stop;
    float elapsed_time;
    
    // Allocate memory for the variables on the host
    complex_data = (cufftComplex *)malloc(size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; ++i) {
        complex_data[i].x = rand() / (float)RAND_MAX;
        complex_data[i].y = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Complex data...\n");
        printf_cufft_cmplx_array(complex_data, element_size);
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

    // -----------------------
    // 1. Perform 3D FFTs over each W slice (W batches of 3D volumes)
    // -----------------------
    CHECK_CUFFT(cufftPlan3d(&plan3d, nx, ny, nz, CUFFT_C2C));
    for (int w = 0; w < nw; ++w) {
        size_t offset = w * nx * ny * nz;
        CHECK_CUFFT(cufftExecC2C(plan3d, d_complex_data + offset, d_complex_data + offset, CUFFT_FORWARD));
    }

    // -----------------------
    // 2. Perform 1D FFT along W dimension
    // -----------------------
    // There are NX*NY*NZ such transforms (one for each (x,y,z) point)
    int n[1] = { (int)nw };
    int batch = nx * ny * nz;
    int embed[1] = { (int)nw };
    int stride = 1;
    int dist = nw;

    CHECK_CUFFT(cufftPlanMany(&plan1d, 1, n,       // rank, dimensions
                                embed, stride, dist,
                                embed, stride, dist,
                                CUFFT_C2C, batch));

    // Execute the 1D FFTs (in-place)
    CHECK_CUFFT(cufftExecC2C(plan1d, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_data, d_complex_data, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_cufft_cmplx_array(complex_data, element_size);
    }

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    // printf("%.6f\n", elapsed_time * 1e-3);

    // Cleanup
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan3d));
    CHECK_CUFFT(cufftDestroy(plan1d));
    free(complex_data);

    return elapsed_time * 1e-3;
}


int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Error: This program requires exactly 5 command-line arguments.\n");
        printf("       %s <arg0> <arg1> <arg2> <arg3> <arg4>\n", argv[0]);
        printf("       arg0, arg1, arg2, arg3: FFT lengths in 4D\n");
        printf("       arg4: Number of iterations\n");
        printf("       e.g.: %s 64 64 64 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int nz = atoi(argv[3]);
    unsigned int nw = atoi(argv[4]);
    unsigned int niter = atoi(argv[5]);

    // Discard the first time running. It apparantly does some extra work during first time
    // JIT??
    run_test_cufft_4d_3d1d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_cufft_4d_3d1d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}