# my-cuda-playground

## Run NVCC
nvcc -o build/module src/module.cu --ptxas-options=-v --use_fast_math -lcufft

## Run GCC
gcc -o build/module src/module.c  -lfftw3 -lm
