#include <mpi.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 64
#define NY 64
#define NZ 512
#define NW 128

void printf_fftw_cmplx_array(fftw_complex *complex_array, unsigned int size) {
        for (unsigned int i = 0; i < 5; ++i) {
            printf("  (%2.4f, %2.4fi)\n", complex_array[i][0], complex_array[i][1]);
        }
        printf("...\n");
        for (unsigned int i = size - 5; i < size; ++i) {
            printf("  (%2.4f, %2.4fi)\n", complex_array[i][0], complex_array[i][1]);
        }
        // for (unsigned int i = 0; i < size; ++i) {
        //         printf("  (%2.4f, %2.4fi)\n", complex_array[i][0], complex_array[i][1]);
        // }
    }

int main (int argv, char **argc) {
        srand(2025);
	int rank, size;	
	fftw_complex *complex_data;
	double t1, t2;
	fftw_plan plan;
	ptrdiff_t alloc_local, local_n0, local_0_start;
        ptrdiff_t nn[4] = {NX, NY, NZ, NW};

	MPI_Init (&argv, &argc);
	fftw_mpi_init ();
	
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);

        // printf("Rank: %d | Proc: %d\n", rank, size);
	
	alloc_local = fftw_mpi_local_size (4, nn, 
                                          MPI_COMM_WORLD, 
                                          &local_n0,  
                                          &local_0_start);

        // printf("local_n0: %ld | local_0_start: %ld\n", local_n0, local_0_start);
        // printf("alloc_local: %ld\n", alloc_local);
	
	complex_data = fftw_alloc_complex (alloc_local);

	plan = fftw_mpi_plan_dft (4, nn, complex_data, complex_data, 
				 MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

	// Initialize input with some numbers	
	for (int i = 0; i < local_n0; i++) {
		for (int j = 0; j < NY * NZ * NW; j++) {
                        complex_data[i*NX + j][0] = rand() / (float)RAND_MAX;
                        complex_data[i*NX + j][1] = 0;
                }
        }

        // for (unsigned int i = 0; i < NX * NY * NZ * NW; ++i) {
        //         complex_data[i][0] = rand() / (float)RAND_MAX;
        //         complex_data[i][1] = 0;
        // }

        // printf("Complex data...\n");
        // printf_fftw_cmplx_array(complex_data, NX * NY * NZ * NW);

        for (int i = 0; i < 100; ++i ) {
                // Start the clock
                MPI_Barrier (MPI_COMM_WORLD);
                t1 = MPI_Wtime ();
                
                // Do a fourier transform
                fftw_execute(plan);

                // Stop the clock
                MPI_Barrier (MPI_COMM_WORLD);
                t2 = MPI_Wtime ();

                // Print out how long it took in seconds
                if (rank == 0) printf("Loop time is %gs with %d procs\n", t2-t1, size);
        }

        // printf("FFT Coefficients...\n");
        // printf_fftw_cmplx_array(complex_data, NX * NY * NZ * NW);

	// Clean up and get out
	fftw_free (complex_data);
	fftw_destroy_plan (plan);
	MPI_Finalize ();
	return 0;
}