#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define NX 64
#define NY 64
#define NZ 32
#define NW 1024

float run_kmb_cufft_4d_3d1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    cufftComplex *h_complex_data, *d_complex_data;
    cufftHandle plan1d_w, plan3d_xyz;

    cudaEvent_t start, stop;
    float elapsed_time;

    unsigned int element_size = NX * NY * NZ * NW;
    size_t size = sizeof(cufftComplex) * element_size;

    // Allocate memory for the variables on the host
    h_complex_data = (cufftComplex *)malloc(size);

    // Create CUDA events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Allocate device memory for complex signal and output frequency
    CHECK_CUDA(cudaMalloc((void **)&d_complex_data, size));

    // Setup FFT plans
    int n_w[1] = { (int)NW };
    CHECK_CUFFT(cufftPlanMany(&plan1d_w, 1, n_w,       // 1D FFT of size nw
                            NULL, 1, NW, // inembed, istride, idist
                            NULL, 1, NW, // onembed, ostride, odist
                            CUFFT_C2C, NX * NY * NZ));
    int n_xyz[3] = { (int)NX, (int)NY, (int)NZ };
    int embed[3] = { (int)NX, (int)NY, (int)NZ };
    CHECK_CUFFT(cufftPlanMany(&plan3d_xyz, 3, n_xyz,          // 3D FFT of size nx*ny*nz
                            embed, NW/2, 1,     // inembed, istride, idist
                            embed, NW/2, 1,     // onembed, ostride, odist
                            CUFFT_C2C, NW/2));

    // 1. Generate a 4D dataset
    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; ++i) {
        h_complex_data[i].x = rand() / (float)RAND_MAX;
        h_complex_data[i].y = 0;
    }
    
    // 1.1. Zero pad the matrix
    // https://stackoverflow.com/questions/69536258/cuda-zeropadding-3d-matrix

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_complex_data, h_complex_data, size, cudaMemcpyHostToDevice));

    // 2. Perform the 1D FFT along the 4th dimension
    CHECK_CUFFT(cufftExecC2C(plan1d_w, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // 3. Perform the 3D FFT along the rest of the dimensions
    CHECK_CUFFT(cufftExecC2C(plan3d_xyz, d_complex_data, d_complex_data, CUFFT_FORWARD));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_complex_data, d_complex_data, size, cudaMemcpyDeviceToHost));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Cleanup
    CHECK_CUFFT(cufftDestroy(plan1d_w));
    CHECK_CUFFT(cufftDestroy(plan3d_xyz));
    CHECK_CUDA(cudaFree(d_complex_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_complex_data);

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
    run_kmb_cufft_4d_3d1d(nx, ny, nz, nw);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_kmb_cufft_4d_3d1d(nx, ny, nz, nw);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}