#include "../../common/common.h"
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

__global__ void extract_xyz_slice(cufftComplex* d_out, cufftComplex* d_in, int nx, int ny, int nz, int nw, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        int in_idx  = (((x * ny + y) * nz + z) * nw) + w;
        int out_idx = ((x * ny + y) * nz + z);
        d_out[out_idx] = d_in[in_idx];
    }
}

__global__ void write_xyz_slice_back(cufftComplex* d_out, cufftComplex* d_in, int nx, int ny, int nz, int nw, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        int out_idx = (((x * ny + y) * nz + z) * nw) + w;
        int in_idx  = ((x * ny + y) * nz + z);
        d_out[out_idx] = d_in[in_idx];
    }
}

float run_test_cufft_4d_3d1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);
    
    // Declaration
    cufftComplex *complex_data, *d_complex_data;
    cufftComplex *d_temp3d_xyz;
    cufftHandle plan3d_xyz, plan1d_w;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    unsigned int element_size_xyz = nx * ny * nz;
    size_t size_xyz = sizeof(cufftComplex) * element_size_xyz;

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

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    // Setup FFT plans
    CHECK_CUDA(cudaMalloc((void **)&d_temp3d_xyz, size_xyz));
    CHECK_CUFFT(cufftPlan3d(&plan3d_xyz, nx, ny, nz, CUFFT_C2C));
    int n[1] = { (int)nw };
    CHECK_CUFFT(cufftPlanMany(&plan1d_w, 1, n,       // 1D FFT of size nw
                            NULL, 1, nw, // inembed, istride, idist
                            NULL, 1, nw, // onembed, ostride, odist
                            CUFFT_C2C, nx * ny * nz));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // -----------------------
    // 1. Perform 3D FFTs over each W slice (W batches of 3D volumes)
    // -----------------------    
    for (int w = 0; w < nw; ++w) {
        // Copy the W-slice into 3D matrices
        extract_xyz_slice<<<blocks, threads>>>(d_temp3d_xyz, d_complex_data, nx, ny, nz, nw, w);
        // Perform 2D FFT
        CHECK_CUFFT(cufftExecC2C(plan3d_xyz, d_temp3d_xyz, d_temp3d_xyz, CUFFT_FORWARD));
        // Copy back
        write_xyz_slice_back<<<blocks, threads>>>(d_complex_data, d_temp3d_xyz, nx, ny, nz, nw, w);
    }

    // -----------------------
    // 2. Perform 1D FFT along W dimension
    // -----------------------
    // Execute FFT on the batched data
    CHECK_CUFFT(cufftExecC2C(plan1d_w, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // Copy results back to host
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
    CHECK_CUFFT(cufftDestroy(plan3d_xyz));
    CHECK_CUFFT(cufftDestroy(plan1d_w));
    CHECK_CUDA(cudaFree(d_temp3d_xyz));
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
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
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_cufft_4d_3d1d(nx, ny, nz, nw);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}