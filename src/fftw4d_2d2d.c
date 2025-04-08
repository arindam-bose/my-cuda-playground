#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define PRINT_FLAG 0
#define NPRINTS 5  // print size

void printf_fftw_cmplx_array(fftw_complex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i][0], complex_array[i][1]);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  (%2.4f, %2.4fi)\n", complex_array[i][0], complex_array[i][1]);
    }
}

float run_test_fftw_4d_2d2d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    fftw_complex *complex_data;
    fftw_complex *temp2d_xy, *temp2d_zw;
    fftw_plan plan2d_xy, plan2d_zw;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(fftw_complex) * element_size;

    unsigned int element_size_xy = nx * ny;
    size_t size_xy = sizeof(fftw_complex) * element_size_xy;
    unsigned int element_size_zw = nz * nw;
    size_t size_zw = sizeof(fftw_complex) * element_size_zw;

    clock_t start, stop;
    float elapsed_time;

    // Allocate memory for input and output arrays
    complex_data = (fftw_complex *)fftw_malloc(size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; ++i) {
        complex_data[i][0] = rand() / (float)RAND_MAX;
        complex_data[i][1] = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Complex data...\n");
        printf_fftw_cmplx_array(complex_data, element_size);
    }
    
    // Start time
    start = clock();

    // ---- 1. 2D FFT over (X, Y) for each Z, W ----
    temp2d_xy = fftw_malloc(size_xy);
    plan2d_xy = fftw_plan_dft_2d(nx, ny, temp2d_xy, temp2d_xy, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int z = 0; z < nz; ++z) {
        for (int w = 0; w < nw; ++w) {
            // Extract 2D slice
            for (int x = 0; x < nx; ++x) {
                for (int y = 0; y < ny; ++y) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    temp2d_xy[x * ny + y][0] = complex_data[idx][0];
                    temp2d_xy[x * ny + y][1] = complex_data[idx][1];
                }
            }
            fftw_execute(plan2d_xy);
            // Copy back
            for (int x = 0; x < nx; ++x) {
                for (int y = 0; y < ny; ++y) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx][0] = temp2d_xy[x * ny + y][0];
                    complex_data[idx][1] = temp2d_xy[x * ny + y][1];
                }
            }
        }
    }

    // ---- 2. 2D FFT over (Z, W) for each X, Y ----
    temp2d_zw = fftw_malloc(size_zw);
    plan2d_zw = fftw_plan_dft_2d(nz, nw, temp2d_zw, temp2d_zw, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            // Extract 2D slice
            for (int z = 0; z < nz; ++z) {
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    temp2d_zw[z * nw + w][0] = complex_data[idx][0];
                    temp2d_zw[z * nw + w][1] = complex_data[idx][1];
                }
            }
            fftw_execute(plan2d_zw);
            // Copy back
            for (int z = 0; z < nz; ++z) {
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx][0] = temp2d_zw[z * nw + w][0];
                    complex_data[idx][1] = temp2d_zw[z * nw + w][1];
                }
            }
        }
    }

    // End time
    stop = clock();

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Output...\n");
        printf_fftw_cmplx_array(complex_data, element_size);
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Clean up
    fftw_destroy_plan(plan2d_xy);
    fftw_destroy_plan(plan2d_zw);
    fftw_free(complex_data);
    fftw_free(temp2d_xy);
    fftw_free(temp2d_zw);
    fftw_cleanup();

    return elapsed_time;
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

    // Discard the first time running for this as well to make apples-to-apples comparison
    run_test_fftw_4d_2d2d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_fftw_4d_2d2d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}
