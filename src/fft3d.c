#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define PRINT_FLAG 0
#define NPRINTS 30  // print size
#define IFFT_FLAG 0

void run_test_fft_3d(unsigned int nx, unsigned int ny, unsigned int nz) {
    srand(2025);

    // Declaration
    fftw_complex *complex_samples, *new_complex_samples;
    fftw_complex *complex_freq;
    fftw_plan plan_fft, plan_ifft;
    
    unsigned int element_size = nx * ny * nz;

    // Allocate memory for input and output arrays
    complex_samples = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * element_size);
    complex_freq = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * element_size);
    if (IFFT_FLAG) {new_complex_samples = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * element_size);}

    // Initialize input complex signal
    for (unsigned int i = 0; i < element_size; i++) {
        complex_samples[i][0] = rand() / (float)RAND_MAX;
        complex_samples[i][1] = 0;
    }

    // Setup the FFT plan
    plan_fft = fftw_plan_dft_3d(nx, ny, nz, complex_samples, complex_freq, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute a complex-to-complex 3D FFT
    fftw_execute(plan_fft);

    if (IFFT_FLAG) {
        // Setup the IFFT plan
        plan_ifft = fftw_plan_dft_3d(nx, ny, nz, complex_freq, new_complex_samples, FFTW_BACKWARD, FFTW_ESTIMATE);

        // Execute a complex-to-complex 3D IFFT
        fftw_execute(plan_ifft);

        // Normalize
        for (unsigned int i = 0; i < element_size; i++) {
            new_complex_samples[i][0] /= (float)element_size;
            new_complex_samples[i][1] /= (float)element_size;
        }
    }

    // Print output stuff
    if (PRINT_FLAG && IFFT_FLAG) {
        printf("Complex samples after FFT and IFFT...\n");
        for (unsigned int i = 0; i < NPRINTS; i++) {
            if (IFFT_FLAG)
            printf("  %2.4f + i%2.4f -> %2.4f + i%2.4f\n", complex_samples[i][0], complex_samples[i][1], new_complex_samples[i][0], new_complex_samples[i][1]);
        }
    }

    // Cleanups
    fftw_destroy_plan(plan_fft);
    fftw_destroy_plan(plan_ifft);
    if (IFFT_FLAG) {fftw_free(new_complex_samples);}
    fftw_free(complex_samples);
    fftw_free(complex_freq);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Error: This program requires exactly 3 command-line arguments.\n");
        return 1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    run_test_fft_3d(nx, ny, nz);
    return 0;
}