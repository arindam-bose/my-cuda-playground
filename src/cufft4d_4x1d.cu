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

// Function to execute 1D FFT along a specific dimension
// void execute_cufft1d(cufftComplex *d_idata, cufftComplex *d_odata, int dim_size, int batch, int stride, int dist) {
//     cufftHandle plan;
//     CHECK_CUFFT(cufftPlanMany(&plan, 1, &dim_size, 
//                                 NULL, stride, dist, 
//                                 NULL, stride, dist, 
//                                 CUFFT_C2C, batch));

//     // Perform FFT
//     CHECK_CUFFT(cufftExecC2C(plan, d_idata, d_odata, CUFFT_FORWARD));
//     CHECK_CUFFT(cufftDestroy(plan));
// }

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
    cufftComplex *complex_samples;
    cufftComplex *complex_freq;
    cufftComplex *d_complex_samples;
    cufftComplex *d_complex_freq;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(cufftComplex) * element_size;

    cudaEvent_t start, stop;
    float elapsed_time;

    // Allocate memory for the variables on the host
    complex_samples = (cufftComplex *)malloc(size);
    complex_freq = (cufftComplex *)malloc(size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; ++i) {
        complex_samples[i].x = rand() / (float)RAND_MAX;
        complex_samples[i].y = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Complex data...\n");
        printf_cufft_cmplx_array(complex_samples, element_size);
    }

    // Create CUDA events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_samples, size));
    CHECK_CUDA(cudaMalloc((void **)&d_complex_freq, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_samples, complex_samples, size, cudaMemcpyHostToDevice));

    // Perform FFT along each dimension sequentially
    // execute_cufft1d(d_complex_samples, d_complex_freq, nx, ny * nz * nw, 1, nx);         // FFT along X
    // execute_cufft1d(d_complex_freq, d_complex_freq, ny, nx * nz * nw, nx, ny);           // FFT along Y
    // execute_cufft1d(d_complex_freq, d_complex_freq, nz, nx * ny * nw, nx * ny, nz);      // FFT along Z
    // execute_cufft1d(d_complex_freq, d_complex_freq, nw, nx * ny * nz, nx * ny * nz, nw); // FFT along W
    // int n[1];           // FFT length
    // int stride = 1;    
    // int dist;
    // int embed[1];

    // n[0] = nx; embed[0] = nx * ny * nz * nw; dist = ny * nz * nw;
    // execute_cufft1d(d_complex_samples, d_complex_freq, n, embed, stride, dist, 1);                         // FFT along X
    // n[0] = ny; embed[0] = ny * nz * nw; dist = nz * nw;
    // execute_cufft1d(d_complex_freq, d_complex_freq, n, embed, stride, dist, nx);                        // FFT along Y
    // n[0] = nz; embed[0] = nz * nw; dist = nw;
    // execute_cufft1d(d_complex_freq, d_complex_freq, n, embed, stride, dist, nx * ny);                   // FFT along Z
    // n[0] = nw; embed[0] = nw; dist = 1;
    // execute_cufft1d(d_complex_freq, d_complex_freq, n, embed, stride, dist, nx * ny * nz);           // FFT along W


    cufftHandle plan;
    int n[4] = {nw, nz, ny, nx};
    CHECK_CUFFT(cufftPlanMany(
        &plan,
        4, // Number of dimensions
        n, // Dimensions of the array (in reverse order for CUFFT)
        NULL, 1, 0,             // Input strides and embed
        NULL, 1, 0,             // Output strides and embed
        CUFFT_C2C,              // Transform type: complex-to-complex
        1                       // Number of transforms (batch size)
    ));

    // Execute the forward 4D FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_complex_samples, d_complex_freq, CUFFT_FORWARD));


    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex_freq, d_complex_freq, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_cufft_cmplx_array(complex_freq, element_size);
    }

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    // printf("%.6f\n", elapsed_time * 1e-3);

    // Clean up
    CHECK_CUDA(cudaFree(d_complex_freq));
    CHECK_CUDA(cudaFree(d_complex_samples));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(complex_freq);
    free(complex_samples);

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