#!/bin/bash

# For NVCC
# Create the executable
PROGRAM=cufft3d
# PROGRAM=fftw3d

nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
# gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm

# Define argument sets (each line represents a set of arguments)
ARGS_LIST=(
    "8 8 8"
    "16 16 16"
    "32 32 32"
    "64 64 64"
    "128 128 128"
    "256 256 256"
    "512 512 512"
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