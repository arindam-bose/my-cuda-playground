#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define PRINT_FLAG 1
#define NPRINTS 5  // print size

void printf_cufft_cmplx_array(cufftComplex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i].x, complex_array[i].y);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i].x, complex_array[i].y);
    }
}

float run_test_cufft_4d_2d2d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;
    cufftHandle plan_xy, plan_zw;

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

    // -----------------------------
    // 1. 2D FFT in (X, Y) direction
    // -----------------------------
    // We'll perform NX x NY FFTs for each (Z,W) slice => NZ*NW batches
    int n_xy[2] = { (int)nx, (int)ny };
    int batch_xy = nz * nw;

    int inembed_xy[2] = { (int)nx, (int)ny };
    int onembed_xy[2] = { (int)nx, (int)ny };
    int istride_xy = 1;
    int ostride_xy = 1;
    int idist_xy = nx * ny;
    int odist_xy = nx * ny;

    CHECK_CUFFT(cufftPlanMany(&plan_xy, 2, n_xy,
                              inembed_xy, istride_xy, idist_xy,
                              onembed_xy, ostride_xy, odist_xy,
                              CUFFT_C2C, batch_xy));

    CHECK_CUFFT(cufftExecC2C(plan_xy, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // ----------------------------------
    // 2. 2D FFT in (Z, W) direction
    // ----------------------------------
    // We need to reinterpret the data layout: flatten (Z,W) for each (X,Y)
    // We perform NZ x NW FFTs for each (X,Y) location => NX*NY batches
    int n_zw[2] = { (int)nz, (int)nw };
    int batch_zw = nx * ny;

    int inembed_zw[2] = { (int)nz, (int)nw };
    int onembed_zw[2] = { (int)nz, (int)nw };
    int istride_zw = 1;
    int ostride_zw = 1;
    int idist_zw = nz * nw;
    int odist_zw = nz * nw;

    // Note: You may need to rearrange your data if not stored in the proper order.
    CHECK_CUFFT(cufftPlanMany(&plan_zw, 2, n_zw,
                              inembed_zw, istride_zw, idist_zw,
                              onembed_zw, ostride_zw, odist_zw,
                              CUFFT_C2C, batch_zw));

    CHECK_CUFFT(cufftExecC2C(plan_zw, d_complex_data, d_complex_data, CUFFT_FORWARD));

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

    // Cleanup
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan_xy));
    CHECK_CUFFT(cufftDestroy(plan_zw));
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
    run_test_cufft_4d_2d2d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_cufft_4d_2d2d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}