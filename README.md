# Cuda Playground

## Installation
```
sudo install python3-virtualenv
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xvzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure
make
sudo make install

git clone https://github.com/arindam-bose/my-cuda-playground
cd my-python-playground
virtualenv venv -p /usr/bin/python3
source venv/bin/activate
pip install -r requirements.txt
mkdir build
```
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

## Run NVCC individual program
```nvcc -o build/module src/module.cu --ptxas-options=-v --use_fast_math -lcufft```

## Run GCC individual program
```gcc -o build/module src/module.c -lfftw3 -lm```

python ./scripts/plot_et.py 
