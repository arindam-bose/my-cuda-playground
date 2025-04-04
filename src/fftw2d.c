#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_fftw_2d(unsigned int nx, unsigned int ny) {
    srand(2025);

    // Declaration
    fftw_complex *complex_samples;
    fftw_complex *complex_freq;
    fftw_plan plan;

    unsigned int element_size = nx * ny;
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
        for (unsigned int i = 0; i < NPRINTS; ++i) {
            printf("  %2.4f + i%2.4f\n", complex_samples[i][0], complex_samples[i][1]);
        }
    }

    // Start time
    start = clock();

    // Setup the FFT plan
    plan = fftw_plan_dft_2d(nx, ny, complex_samples, complex_freq, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute a complex-to-complex 2D FFT
    fftw_execute(plan);

    // End time
    stop = clock();

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        for (unsigned int i = 0; i < NPRINTS; ++i) {
            printf("  %2.4f + i%2.4f\n", complex_freq[i][0], complex_freq[i][1]);
        }
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("%.6f\n", elapsed_time);

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(complex_samples);
    fftw_free(complex_freq);
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Error: This program requires exactly 2 command-line arguments.\n");
        printf("       %s <arg0> <arg1>\n", argv[0]);
        printf("       arg0, arg1: FFT lengths in 2D\n");
        printf("       e.g.: %s 64 64\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    run_test_fftw_2d(nx, ny);
    return 0;
}