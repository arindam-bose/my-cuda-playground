# my-cuda-playground

## Versions
### NVCC
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
```

### GCC
```gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0```

### FFTW
```FFTW 3.3.10```


## Run NVCC
```nvcc -o build/module src/module.cu --ptxas-options=-v --use_fast_math -lcufft```

## Run GCC
```gcc -o build/module src/module.c  -lfftw3 -lm```
