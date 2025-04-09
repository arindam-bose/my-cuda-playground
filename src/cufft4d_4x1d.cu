#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

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

void execute_cufft1d(cufftComplex *d_idata, cufftComplex *d_odata, int *dim, int *embed, int stride, int dist, int batch) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(&plan, 1, dim, 
                                embed, stride, dist, 
                                embed, stride, dist, 
                                CUFFT_C2C, batch));

    // Perform FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_idata, d_odata, CUFFT_FORWARD));
    CHECK_CUFFT(cufftDestroy(plan));
}

float run_test_cufft_4d_4x1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *complex_data;
    cufftComplex *d_complex_data;

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

    cufftHandle plan1d_x, plan1d_y, plan1d_z, plan1d_w;
    int n[1] = { (int)nx };
    CHECK_CUFFT(cufftPlanMany(&plan1d_x, 1, n,       // 1D FFT of size nx
                            NULL, ny * nz * nw, nz, // inembed, istride, idist
                            NULL, ny * nz * nw, nx, // onembed, ostride, odist
                            CUFFT_C2C, ny * nz * nw));
    n[0] = (int)ny;
    CHECK_CUFFT(cufftPlanMany(&plan1d_y, 1, n,       // 1D FFT of size ny
                            NULL, nz * nw, ny, // inembed, istride, idist
                            NULL, nz * nw, ny, // onembed, ostride, odist
                            CUFFT_C2C, nx * nz * nw));
    n[0] = (int)nz;
    CHECK_CUFFT(cufftPlanMany(&plan1d_z, 1, n,       // 1D FFT of size nz
                            NULL, nw, nz, // inembed, istride, idist
                            NULL, nw, nz, // onembed, ostride, odist
                            CUFFT_C2C, nx * ny * nw));
    n[0] = (int)nw;
    CHECK_CUFFT(cufftPlanMany(&plan1d_w, 1, n,       // 1D FFT of size nw
                            NULL, 1, nw, // inembed, istride, idist
                            NULL, 1, nw, // onembed, ostride, odist
                            CUFFT_C2C, nx * ny * nz));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, complex_data, size, cudaMemcpyHostToDevice));

    // Perform FFT along each dimension sequentially
    CHECK_CUFFT(cufftExecC2C(plan1d_x, d_complex_data, d_complex_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan1d_y, d_complex_data, d_complex_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan1d_z, d_complex_data, d_complex_data, CUFFT_FORWARD));
    // CHECK_CUFFT(cufftExecC2C(plan1d_w, d_complex_data, d_complex_data, CUFFT_FORWARD));

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
    CHECK_CUFFT(cufftDestroy(plan1d_w));
    CHECK_CUFFT(cufftDestroy(plan1d_z));
    CHECK_CUFFT(cufftDestroy(plan1d_y));
    CHECK_CUFFT(cufftDestroy(plan1d_x));
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
    run_test_cufft_4d_4x1d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_cufft_4d_4x1d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}