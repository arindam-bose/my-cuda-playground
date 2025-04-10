#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define PRINT_FLAG 1
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

float run_test_fftw_3d(unsigned int nx, unsigned int ny, unsigned int nz) {
    srand(2025);

    // Declaration
    fftw_complex *complex_data;
    fftw_plan plan_fft;
    
    unsigned int element_size = nx * ny * nz;
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
    plan_fft = fftw_plan_dft_3d(nx, ny, nz, complex_data, complex_data, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Start time
    start = clock();

    // Execute a complex-to-complex 3D FFT
    fftw_execute(plan_fft);

    // End time
    stop = clock();

    // if (IFFT_FLAG) {
    //     // Setup the IFFT plan
    //     plan_ifft = fftw_plan_dft_3d(nx, ny, nz, complex_freq, new_complex_samples, FFTW_BACKWARD, FFTW_ESTIMATE);

    //     // Execute a complex-to-complex 3D IFFT
    //     fftw_execute(plan_ifft);

    //     // Normalize
    //     for (unsigned int i = 0; i < element_size; ++i) {
    //         new_complex_samples[i][0] /= (float)element_size;
    //         new_complex_samples[i][1] /= (float)element_size;
    //     }
    // }

    // Print output stuff
    // if (PRINT_FLAG && IFFT_FLAG) {
    //     printf("Complex samples after FFT and IFFT...\n");
    //     for (unsigned int i = 0; i < NPRINTS; ++i) {
    //         if (IFFT_FLAG)
    //         printf("  %2.4f + i%2.4f -> %2.4f + i%2.4f\n", complex_samples[i][0], complex_samples[i][1], new_complex_samples[i][0], new_complex_samples[i][1]);
    //     }
    // }
    if (PRINT_FLAG) {
        printf("Fourier Coefficients...\n");
        printf_fftw_cmplx_array(complex_data, element_size);
    }

    // Compute elapsed time
    elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Clean up
    fftw_destroy_plan(plan_fft);
    fftw_free(complex_data);
    fftw_cleanup();

    return elapsed_time;
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Error: This program requires exactly 4 command-line arguments.\n");
        printf("       %s <arg0> <arg1> <arg2> <arg3>\n", argv[0]);
        printf("       arg0, arg1, arg2: FFT lengths in 3D\n");
        printf("       arg3: Number of iterations\n");
        printf("       e.g.: %s 64 64 64 5\n", argv[0]);
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int ny = atoi(argv[2]);
    unsigned int nz = atoi(argv[3]);
    unsigned int niter = atoi(argv[4]);

    // Discard the first time running for this as well to make apples-to-apples comparison
    run_test_fftw_3d(nx, ny, nz);

    float sum = 0.0;
    float span_s = 0.0;
    for (unsigned int i = 0; i < niter; ++i) {
        span_s = run_test_fftw_3d(nx, ny, nz);
        if (PRINT_FLAG) printf("[%d]: %.6f s\n", i, span_s);
        sum += span_s;
    }
    printf("%.6f\n", sum/(float)niter);

    return 0;
}