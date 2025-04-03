#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define N   1048576  // dimension size
#define M_PI   3.14159265358979323846  /* pi */
#define PRINT_FLAG 0
#define NPRINTS 30  // print size

void run_test_fftw_1d(int argc, char** argv) {
    // Declaration
    float *samples;
    fftw_complex *complex_samples;
    fftw_complex *complex_freq;
    fftw_plan plan;

    // Allocate memory for input and output arrays
    samples = (float *)malloc(sizeof(float) * N);
    complex_samples = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    complex_freq = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Input signal generation using cos(x)
    double delta = M_PI / 20.0;
    for (unsigned int i = 0; i < N; i++) {
        samples[i] = cos(i * delta);
    }

    // Convert to a complex signal
    for (unsigned int i = 0; i < N; i++) {
        complex_samples[i][0] = samples[i];
        complex_samples[i][1] = 0;
    }

    // Print input stuff
    if (PRINT_FLAG) {
        printf("Real data...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f\n", samples[i]);
        }
        printf("Complex data...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            printf("  %2.4f + i%2.4f\n", complex_samples[i][0], complex_samples[i][1]);
        }
    }

    // Setup the FFT plan
    plan = fftw_plan_dft_1d(N, complex_samples, complex_freq, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute a complex-to-complex 1D FFT
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
    free(samples);
}


int main(int argc, char **argv) {
    run_test_fftw_1d(argc, argv);
    return 0;
}