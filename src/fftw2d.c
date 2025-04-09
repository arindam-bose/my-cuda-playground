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

float run_test_fftw_2d(unsigned int nx, unsigned int ny) {
    srand(2025);

    // Declaration
    fftw_complex *complex_data;
    fftw_plan plan;

    unsigned int element_size = nx * ny;
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

    // Setup the FFT plan
    plan = fftw_plan_dft_2d(nx, ny, complex_data, complex_data, FFTW_FORWARD, FFTW_ESTIMATE);

    // Start time
    start = clock();

    // Execute a complex-to-complex 2D FFT
    fftw_execute(plan);

    // End time
    stop = clock();

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_fftw_cmplx_array(complex_data, element_size);
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(complex_data);
    fftw_cleanup();

    return elapsed_time;
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Error: This program requires exactly 3 command-line arguments.\n");
        printf("       %s <arg0> <arg1> <arg2>\n", argv[0]);
        printf("       arg0, arg1: FFT lengths in 2D\n");
        printf("       arg1: Number of iterations\n");
        printf("       e.g.: %s 64 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int niter = atoi(argv[3]);

    // Discard the first time running for this as well to make apples-to-apples comparison
    run_test_fftw_2d(nx, ny);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_fftw_2d(nx, ny);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}