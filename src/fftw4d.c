#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_fftw_4d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nw) {
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
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i][0] = rand() / (float)RAND_MAX;
        complex_samples[i][1] = 0;
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
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_freq[i][0], complex_freq[i][1]);
        }
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f s\n", elapsed_time);

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(complex_samples);
    fftw_free(complex_freq);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 4 command-line arguments.\n");
        return 1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int nz = atoi(argv[3]);
    unsigned int nw = atoi(argv[4]);
    run_test_fftw_4d(nx, ny, nz, nw);
    return 0;
}