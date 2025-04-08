#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define PRINT_FLAG 0
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

float run_test_cufft_4d_3d1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);
    
    // Declaration
    cufftComplex *complex_data;
    cufftComplex *temp3d, *temp1d;
    cufftComplex *d_temp3d, *d_temp1d;
    cufftHandle plan3d, plan1d;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    unsigned int element_size_xyz = nx * ny * nz;
    size_t size_xyz = sizeof(cufftComplex) * element_size_xyz;
    size_t size_w = sizeof(cufftComplex) * nw;

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

    // // Allocate device memory for complex signal and output frequency
    // CHECK_CUDA(cudaMalloc((void **)&d_complex_data, size));

    // // Copy host memory to device
    // CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // -----------------------
    // 1. Perform 3D FFTs over each W slice (W batches of 3D volumes)
    // -----------------------
    temp3d = (cufftComplex *)malloc(size_xyz);
    CHECK_CUDA(cudaMalloc((void **)&d_temp3d, size_xyz));
    CHECK_CUFFT(cufftPlan3d(&plan3d, nx, ny, nz, CUFFT_C2C));
    for (int w = 0; w < nw; ++w) {
        // Copy the W-slice into tmp_3d
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int src_idx = (((x * ny + y) * nz + z) * nw) + w;
                    int dst_idx = ((x * ny + y) * nz) + z;
                    temp3d[dst_idx].x = complex_data[src_idx].x;
                    temp3d[dst_idx].y = complex_data[src_idx].y;
                }
            }
        }

        CHECK_CUDA(cudaMemcpy(d_temp3d, temp3d, size_xyz, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecC2C(plan3d, d_temp3d, d_temp3d, CUFFT_FORWARD));
        CHECK_CUDA(cudaMemcpy(temp3d, d_temp3d, size_xyz, cudaMemcpyDeviceToHost));

        // Copy results back to data
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int dst_idx = (((x * ny + y) * nz + z) * nw) + w;
                    int src_idx = ((x * ny + y) * nz) + z;
                    complex_data[dst_idx].x = temp3d[src_idx].x;
                    complex_data[dst_idx].y = temp3d[src_idx].y;
                }
            }
        }
    }

    // -----------------------
    // 2. Perform 1D FFT along W dimension
    // -----------------------
    // There are NX*NY*NZ such transforms (one for each (x,y,z) point)
    temp1d = (cufftComplex *)malloc(size_w);
    CHECK_CUDA(cudaMalloc((void **)&d_temp1d, size_w));
    CHECK_CUFFT(cufftPlan1d(&plan1d, nx, CUFFT_C2C, 1));
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                // Copy W vector
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    temp1d[w].x = complex_data[idx].x;
                    temp1d[w].y = complex_data[idx].y;
                }

                // FFT along W
                CHECK_CUDA(cudaMemcpy(d_temp1d, temp1d, size_w, cudaMemcpyHostToDevice));
                CHECK_CUFFT(cufftExecC2C(plan1d, d_temp1d, d_temp1d, CUFFT_FORWARD));
                CHECK_CUDA(cudaMemcpy(temp1d, d_temp1d, size_w, cudaMemcpyDeviceToHost));

                // Copy back
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx].x = temp1d[w].x;
                    complex_data[idx].y = temp1d[w].y;
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
    CHECK_CUFFT(cufftDestroy(plan3d));
    CHECK_CUFFT(cufftDestroy(plan1d));
    CHECK_CUDA(cudaFree(d_temp3d));
    CHECK_CUDA(cudaFree(d_temp1d));
    free(temp3d);
    free(temp1d);
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