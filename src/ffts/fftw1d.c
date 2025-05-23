#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define M_PI   3.14159265358979323846  /* pi */
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

float run_test_fftw_1d(unsigned int nx) {
    // Declaration
    float *samples;
    fftw_complex *complex_data;
    fftw_plan plan;

    size_t size = sizeof(fftw_complex) * nx;

    clock_t start, stop;
    float elapsed_time;

    // Allocate memory for input and output arrays
    samples = (float *)malloc(sizeof(float) * nx);
    complex_data = (fftw_complex *)fftw_malloc(size);

    // Input signal generation using cos(x)
    double delta = M_PI / 20.0;
    for (unsigned int i = 0; i < nx; ++i) {
        samples[i] = cos(i * delta);
    }

    // Convert to a complex signal
    for (unsigned int i = 0; i < nx; ++i) {
        complex_data[i][0] = samples[i];
        complex_data[i][1] = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Real data...\n");
        for (unsigned int i = 0; i < NPRINTS; ++i) {
            printf("  %2.4f\n", samples[i]);
        }
        printf("Complex data...\n");
        printf_fftw_cmplx_array(complex_data, nx);
    }

    // Setup the FFT plan
    plan = fftw_plan_dft_1d(nx, complex_data, complex_data, FFTW_FORWARD, FFTW_ESTIMATE);

    // Start time
    start = clock();

    // Execute a complex-to-complex 1D FFT
    fftw_execute(plan);

    // End time
    stop = clock();

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_fftw_cmplx_array(complex_data, nx);
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(complex_data);
    free(samples);
    fftw_cleanup();

    return elapsed_time;
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Error: This program requires exactly 2 command-line arguments.\n");
        printf("       %s <arg0> <arg1>\n", argv[0]);
        printf("       arg0: FFT length in 1D\n");
        printf("       arg1: Number of iterations\n");
        printf("       e.g.: %s 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int niter = atoi(argv[2]);

    // Discard the first time running for this as well to make apples-to-apples comparison
    run_test_fftw_1d(nx);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_fftw_1d(nx);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}