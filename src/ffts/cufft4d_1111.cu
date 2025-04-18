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

// Function to execute 1D FFT along a specific dimension
void execute_fft(cufftComplex *d_data, int dim_size, int batch_size) {
    cufftHandle plan;
    int n[1] = { dim_size };
    int embed[1] = { dim_size };
    CHECK_CUFFT(cufftPlanMany(&plan, 1, n, 
                            embed, 1, dim_size, 
                            embed, 1, dim_size, 
                            CUFFT_C2C, batch_size));

    // Perform FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftDestroy(plan));
}


__global__ void do_circular_transpose(cufftComplex *d_out, cufftComplex *d_in, int nx, int ny, int nz, int nw) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        for (int w = 0; w < nw; w++) {
            int in_idx  = ((x * ny + y) * nz + z) * nw + w;
            int out_idx = ((y * nz + z) * nw + w) * nx + x;
            d_out[out_idx] = d_in[in_idx];
        }
    }
}

float run_test_cufft_4d_4x1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;
    cufftComplex *d_complex_data_swap;

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

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_data, size));
    CHECK_CUDA(cudaMalloc((void **)&d_complex_data_swap, size));

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x, (ny + threads.y - 1) / threads.y, (nz + threads.z - 1) / threads.z);

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // Perform FFT along each dimension sequentially
    // Help from: https://forums.developer.nvidia.com/t/3d-and-4d-indexing-4d-fft/12564/2
    // and https://stackoverflow.com/questions/79574267/what-is-the-correct-way-to-perform-4d-fft-in-cuda-by-implementing-1d-fft-in-each

    // step 1: do 1-D FFT along w with number of element nw and batch=nx ny nz
    execute_fft(d_complex_data, nw, nx * ny * nz);
    // step 2: do tranpose operation A(x,y,z,w) → A(y,z,w,x)
    do_circular_transpose<<<blocks, threads>>>(d_complex_data_swap, d_complex_data, nx, ny, nz, nw);
    // step 3: do 1-D FFT along x with number of element nx and batch=n2n3n4
    execute_fft(d_complex_data_swap, nx, ny * nz * nw);
    // step 4: do tranpose operation A(y,z,w,x) → A(z,w,x,y)
    do_circular_transpose<<<blocks, threads>>>(d_complex_data, d_complex_data_swap, ny, nz, nw, nx);
    // step 5: do 1-D FFT along y with number of element ny and batch=n3n4n1
    execute_fft(d_complex_data, ny, nx * nz * nw);
    // step 6: do tranpose operation A(z,w,x,y) → A(w,x,y,z)
    do_circular_transpose<<<blocks, threads>>>(d_complex_data_swap, d_complex_data, nz, nw, nx, ny);
    // step 7: do 1-D FFT along z with number of element nz and batch=n4n1n2
    execute_fft(d_complex_data_swap, nz, nx * ny * nw);
    // step 8: do tranpose operation A(w,x,y,z) → A(x,y,z,w)
    do_circular_transpose<<<blocks, threads>>>(d_complex_data, d_complex_data_swap, nw, nx, ny, nz);

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

    // Clean up
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaFree(d_complex_data_swap));
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
    run_test_cufft_4d_4x1d(nx, ny, nz, nw);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_cufft_4d_4x1d(nx, ny, nz, nw);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}