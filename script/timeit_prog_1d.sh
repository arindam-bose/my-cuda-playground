#!/bin/bash

# Create the executable
# for NVCC
# PROGRAM=cufft1d
# nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft

# for GCC
PROGRAM=fftw1d
gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm

# Define argument sets (each line represents a set of arguments)
ARGS_LIST=(
    "64"
    "128"
    "256"
    "512"
    "1024"
    "2048"
    "4096"
    "8192"
    "16384"
    "32768"
    "65536"
    "131072"
    "262144"
    "524288"
    "1048576"
    "2097152"
    "4194304"
    "8388608"
    "16777216"
    "33554432"
    "67108864"
    "134217728"
    "268435456"
)

# Define the executable name
EXECUTABLE="./build/$PROGRAM"

# Loop through each set of arguments and time the execution
for ARGS in "${ARGS_LIST[@]}"
do
    echo "Running: $EXECUTABLE $ARGS"
    $EXECUTABLE $ARGS
    echo "---------------------------------------"
done