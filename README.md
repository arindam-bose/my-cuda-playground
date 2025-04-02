# my-cuda-playground

nvcc -o build/test_cufft src/test_cufft.cu --ptxas-options=-v --use_fast_math -lcufft
gcc -o build/fft1d_basic src/fft1d_basic.c  -lfftw3 -lm
