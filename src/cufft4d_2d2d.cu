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
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i].x, complex_array[i].y);
    }
}

float run_test_cufft_4d_2d2d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data;
    cufftComplex *temp2d_xy, *temp2d_zw;
    cufftComplex *d_temp2d_xy, *d_temp2d_zw;
    cufftHandle plan2d_xy, plan2d_zw;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    unsigned int element_size_xy = nx * ny;
    size_t size_xy = sizeof(cufftComplex) * element_size_xy;
    unsigned int element_size_zw = nz * nw;
    size_t size_zw = sizeof(cufftComplex) * element_size_zw;

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

    // -----------------------------
    // 1. 2D FFT in (X, Y) direction
    // -----------------------------
    // We'll perform NX x NY FFTs for each (Z,W) slice => NZ*NW batches
    temp2d_xy = (cufftComplex *)malloc(size_xy);
    CHECK_CUDA(cudaMalloc((void **)&d_temp2d_xy, size_xy));
    CHECK_CUFFT(cufftPlan2d(&plan2d_xy, nx, ny, CUFFT_C2C));
    for (int z = 0; z < nz; z++) {
        for (int w = 0; w < nw; w++) {
            // Extract 2D slice
            for (int x = 0; x < nx; x++) {
                for (int y = 0; y < ny; y++) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    temp2d_xy[x * ny + y].x = complex_data[idx].x;
                    temp2d_xy[x * ny + y].y = complex_data[idx].y;
                }
            }
            
            CHECK_CUDA(cudaMemcpy(d_temp2d_xy, temp2d_xy, size_xy, cudaMemcpyHostToDevice));
            CHECK_CUFFT(cufftExecC2C(plan2d_xy, d_temp2d_xy, d_temp2d_xy, CUFFT_FORWARD));
            CHECK_CUDA(cudaMemcpy(temp2d_xy, d_temp2d_xy, size_xy, cudaMemcpyDeviceToHost));
            
            // Copy back
            for (int x = 0; x < nx; x++) {
                for (int y = 0; y < ny; y++) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx].x = temp2d_xy[x * ny + y].x;
                    complex_data[idx].y = temp2d_xy[x * ny + y].y;
                }
            }
        }
    }

    // ----------------------------------
    // 2. 2D FFT in (Z, W) direction
    // ----------------------------------
    // We need to reinterpret the data layout: flatten (Z,W) for each (X,Y)
    // We perform NZ x NW FFTs for each (X,Y) location => NX*NY batches
    temp2d_zw = (cufftComplex *)malloc(size_zw);
    CHECK_CUDA(cudaMalloc((void **)&d_temp2d_zw, size_zw));
    CHECK_CUFFT(cufftPlan2d(&plan2d_zw, nz, nw, CUFFT_C2C));
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            // Extract 2D slice
            for (int z = 0; z < nz; z++) {
                for (int w = 0; w < nw; w++) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    temp2d_zw[z * nw + w].x = complex_data[idx].x;
                    temp2d_zw[z * nw + w].y = complex_data[idx].y;
                }
            }
            
            CHECK_CUDA(cudaMemcpy(d_temp2d_zw, temp2d_zw, size_zw, cudaMemcpyHostToDevice));
            CHECK_CUFFT(cufftExecC2C(plan2d_zw, d_temp2d_zw, d_temp2d_zw, CUFFT_FORWARD));
            CHECK_CUDA(cudaMemcpy(temp2d_zw, d_temp2d_zw, size_zw, cudaMemcpyDeviceToHost));
            
            // Copy back
            for (int z = 0; z < nz; z++) {
                for (int w = 0; w < nw; w++) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx].x = temp2d_zw[z * nw + w].x;
                    complex_data[idx].y = temp2d_zw[z * nw + w].y;
                }
            }
        }
    }

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
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan2d_xy));
    CHECK_CUFFT(cufftDestroy(plan2d_zw));
    CHECK_CUDA(cudaFree(d_temp2d_xy));
    CHECK_CUDA(cudaFree(d_temp2d_zw));
    free(temp2d_xy);
    free(temp2d_zw);
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