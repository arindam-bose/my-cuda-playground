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

__global__ void extract_xy_slice(cufftComplex* d_in, cufftComplex* d_out, int nx, int ny, int nz, int nw, int z, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        int input_idx = (((x * ny + y) * nz + z) * nw) + w;
        int output_idx = x * ny + y;
        d_out[output_idx].x = d_in[input_idx].x;
        d_out[output_idx].y = d_in[input_idx].y;
    }
}

__global__ void extract_zw_slice(cufftComplex* d_in, cufftComplex* d_out, int nx, int ny, int nz, int nw, int x, int y) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (z < nz && w < nw) {
        int input_idx = (((x * ny + y) * nz + z) * nw) + w;
        int output_idx = z * nw + w;
        d_out[output_idx] = d_in[input_idx];
    }
}

__global__ void write_xy_slice_back(cufftComplex* d_out, cufftComplex* d_in, int nx, int ny, int nz, int nw, int z, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        int output_idx = (((x * ny + y) * nz + z) * nw) + w;
        int input_idx = x * ny + y;
        d_out[output_idx].x = d_in[input_idx].x;
        d_out[output_idx].y = d_in[input_idx].y;
    }
}

__global__ void write_zw_slice_back(cufftComplex* d_out, cufftComplex* d_in, int nx, int ny, int nz, int nw, int x, int y) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (z < nz && w < nw) {
        int output_idx = (((x * ny + y) * nz + z) * nw) + w;
        int input_idx = z * nw + w;
        d_out[output_idx] = d_in[input_idx];
    }
}

float run_test_cufft_4d_2d2d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data, *d_complex_data;
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

    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

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
    CHECK_CUDA(cudaMalloc((void **)&d_temp2d_xy, size_xy));
    CHECK_CUFFT(cufftPlan2d(&plan2d_xy, nx, ny, CUFFT_C2C));
    for (int z = 0; z < nz; ++z) {
        for (int w = 0; w < nw; ++w) {
            // Extract 2D slice
            extract_xy_slice<<<blocks, threads>>>(d_complex_data, d_temp2d_xy, nx, ny, nz, nw, z, w);
            CHECK_CUFFT(cufftExecC2C(plan2d_xy, d_temp2d_xy, d_temp2d_xy, CUFFT_FORWARD));
            // Copy back
            write_xy_slice_back<<<blocks, threads>>>(d_complex_data, d_temp2d_xy, nx, ny, nz, nw, z, w);
        }
    }

    // ----------------------------------
    // 2. 2D FFT in (Z, W) direction
    // ----------------------------------
    // We need to reinterpret the data layout: flatten (Z,W) for each (X,Y)
    // We perform NZ x NW FFTs for each (X,Y) location => NX*NY batches
    CHECK_CUDA(cudaMalloc((void **)&d_temp2d_zw, size_zw));
    CHECK_CUFFT(cufftPlan2d(&plan2d_zw, nz, nw, CUFFT_C2C));
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            // Extract 2D slice
            extract_zw_slice<<<blocks, threads>>>(d_complex_data, d_temp2d_zw, nx, ny, nz, nw, x, y);
            CHECK_CUFFT(cufftExecC2C(plan2d_zw, d_temp2d_zw, d_temp2d_zw, CUFFT_FORWARD));            
            // Copy back
            write_zw_slice_back<<<blocks, threads>>>(d_complex_data, d_temp2d_zw, nx, ny, nz, nw, x, y);
        }
    }

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
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan2d_xy));
    CHECK_CUFFT(cufftDestroy(plan2d_zw));
    CHECK_CUDA(cudaFree(d_temp2d_xy));
    CHECK_CUDA(cudaFree(d_temp2d_zw));
    CHECK_CUDA(cudaFree(d_complex_data));
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