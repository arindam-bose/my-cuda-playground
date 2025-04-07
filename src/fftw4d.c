#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define PRINT_FLAG 1
#define NPRINTS 5  // print size

void printf_fftw_cmplx_array(fftw_complex *complex_array, unsigned int size) {
    for (unsigned int i = 0; i < NPRINTS; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i][0], complex_array[i][1]);
    }
    printf("...\n");
    for (unsigned int i = size - NPRINTS; i < size; ++i) {
        printf("  %2.4f + i%2.4f\n", complex_array[i][0], complex_array[i][1]);
    }
}

float run_test_fftw_4d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
    srand(2025);

    // Declaration
    fftw_complex *complex_samples;
    fftw_complex *complex_freq;
    fftw_plan plan;

    unsigned int element_size = nx * ny * nz * nw;
    size_t size = sizeof(fftw_complex) * element_size;

    clock_t start, stop;
    float elapsed_time;

    // Allocate memory for input and output arrays
    complex_samples = (fftw_complex *)fftw_malloc(size);
    complex_freq = (fftw_complex *)fftw_malloc(size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; ++i) {
        complex_samples[i][0] = rand() / (float)RAND_MAX;
        complex_samples[i][1] = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Complex data...\n");
        printf_fftw_cmplx_array(complex_samples, element_size);
    }
    
    // Start time
    start = clock();

    // Setup the FFT plan
    plan = fftw_plan_dft(4, (int[]){nx, ny, nz, nw}, complex_samples, complex_freq, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(plan);

    // End time
    stop = clock();

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_fftw_cmplx_array(complex_freq, element_size);
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(complex_samples);
    fftw_free(complex_freq);

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
    run_test_fftw_4d(nx, ny, nz, nw);

    float sum = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        sum += run_test_fftw_4d(nx, ny, nz, nw);
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}