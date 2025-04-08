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

float run_test_fftw_4d_3d1d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    fftw_complex *complex_data;
    fftw_complex *tmp_3d, *tmp_1d;
    fftw_plan plan3d, plan1d;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(fftw_complex) * element_size;

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

    // -------- 1. Perform 3D FFT for each W slice --------
    tmp_3d = fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
    plan3d = fftw_plan_dft_3d(nx, ny, nz, tmp_3d, tmp_3d, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int w = 0; w < nw; ++w) {
        // Copy the W-slice into tmp_3d
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int src_idx = (((x * ny + y) * nz + z) * nw) + w;
                    int dst_idx = ((x * ny + y) * nz) + z;
                    tmp_3d[dst_idx][0] = complex_data[src_idx][0];
                    tmp_3d[dst_idx][1] = complex_data[src_idx][1];
                }
            }
        }
        fftw_execute(plan3d);
        // Copy results back to data
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int dst_idx = (((x * ny + y) * nz + z) * nw) + w;
                    int src_idx = ((x * ny + y) * nz) + z;
                    complex_data[dst_idx][0] = tmp_3d[src_idx][0];
                    complex_data[dst_idx][1] = tmp_3d[src_idx][1];
                }
            }
        }
    }

    // -------- 2. Perform 1D FFT along W for each (x,y,z) --------
    tmp_1d = fftw_malloc(sizeof(fftw_complex) * nw);
    plan1d = fftw_plan_dft_1d(nw, tmp_1d, tmp_1d, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                // Copy W vector
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    tmp_1d[w][0] = complex_data[idx][0];
                    tmp_1d[w][1] = complex_data[idx][1];
                }

                // FFT along W
                fftw_execute(plan1d);
                // Copy back
                for (int w = 0; w < nw; ++w) {
                    int idx = (((x * ny + y) * nz + z) * nw) + w;
                    complex_data[idx][0] = tmp_1d[w][0];
                    complex_data[idx][1] = tmp_1d[w][1];
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

    fftw_destroy_plan(plan3d);
    fftw_destroy_plan(plan1d);
    fftw_free(tmp_3d);
    fftw_free(tmp_1d);
    fftw_free(complex_data);
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
    run_test_fftw_4d_3d1d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_fftw_4d_3d1d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}