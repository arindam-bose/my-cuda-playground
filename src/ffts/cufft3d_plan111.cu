#include "../../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define PRINT_FLAG 1
#define NPRINTS 16  // print size

void printf_cufft_cmplx_array(cufftComplex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
}

float run_test_cufft_3d(unsigned int nx, unsigned int ny, unsigned int nz) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;
    cufftHandle plan1d_x, plan1d_y, plan1d_z;

    unsigned int element_size = nx * ny * nz;
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

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_data, size));

    // Setup the CUFFT plans
    int n[1] = { (int)nx };
    int embed[1] = { (int)nx };
    CHECK_CUFFT(cufftPlanMany(&plan1d_x, 1, n,          // 1D FFT of size nw
                            embed, ny * nz, 1,      // inembed, istride, idist
                            embed, ny * nz, 1,      // onembed, ostride, odist
                            CUFFT_C2C, ny * 1));
    n[0] = (int)ny;
    embed[0] = (int)ny;
    CHECK_CUFFT(cufftPlanMany(&plan1d_y, 1, n,          // 1D FFT of size nw
                            embed, nz, 1,      // inembed, istride, idist
                            embed, nz, 1,      // onembed, ostride, odist
                            CUFFT_C2C, nx * nz));
    n[0] = (int)nz;
    embed[0] = (int)nz;
    CHECK_CUFFT(cufftPlanMany(&plan1d_z, 1, n,          // 1D FFT of size nw
                            embed, 1, nz,      // inembed, istride, idist
                            embed, 1, nz,      // onembed, ostride, odist
                            CUFFT_C2C, nx * ny));
    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // Execute the forward 3D FFT (in-place computation)
    CHECK_CUFFT(cufftExecC2C(plan1d_x, d_complex_data, d_complex_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan1d_y, d_complex_data, d_complex_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan1d_z, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_data, d_complex_data, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_cufft_cmplx_array(complex_data, element_size);
    }

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan1d_x));
    CHECK_CUFFT(cufftDestroy(plan1d_y));
    CHECK_CUFFT(cufftDestroy(plan1d_z));
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(complex_data);

    return elapsed_time * 1e-3;
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 4 command-line arguments.\n");
        printf("       %s <arg0> <arg1> <arg2> <arg3>\n", argv[0]);
        printf("       arg0, arg1, arg2: FFT lengths in 3D\n");
        printf("       arg3: Number of iterations\n");
        printf("       e.g.: %s 64 64 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int nz = atoi(argv[3]);
    unsigned int niter = atoi(argv[4]);

    // Discard the first time running. It apparantly does some extra work during first time
    // JIT??
    run_test_cufft_3d(nx, ny, nz);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_cufft_3d(nx, ny, nz);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}