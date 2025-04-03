#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

    // Allocate memory for input and output arrays
    complex_samples = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * element_size);
    complex_freq = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * element_size);

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i][0] = rand() / (float)RAND_MAX;
        complex_samples[i][1] = 0;
    }

    // Setup the FFT plan
    plan = fftw_plan_dft(4, (int[]){nx, ny, nz, nw}, complex_samples, complex_freq, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(plan);

    // Print output stuff
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_freq[i][0], complex_freq[i][1]);
        }
    }

    // Cleanups
    fftw_destroy_plan(plan);
    fftw_free(complex_samples);
    fftw_free(complex_freq);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 5 command-line arguments.\n");
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    int nw = atoi(argv[4]);
    run_test_fftw_4d(nx, ny, nz, nw);
    return 0;
}