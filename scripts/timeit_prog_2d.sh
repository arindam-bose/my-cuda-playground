#!/bin/bash

# Create the executable if the arguments are correct
if [ -z "$1" ]; then
    echo "This script should be called with atleast one argument:" >&2
    echo "e.g. $0 cufft" >&2
    echo "     $0 fftw" >&2
    exit 2
elif [ "$1" == "cufft" ]; then
    echo "Generating binary for cufft"
    PROGRAM=cufft2d
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "fftw" ]; then
    echo "Generating binary for fftw"
    PROGRAM=fftw2d
    gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm
else
    echo "First argument should be {cufft|fftw}"
    exit 2
fi

# Define argument sets (each line represents a set of arguments)
ARGS_LIST=(
    "8 8"
    "16 16"
    "32 32"
    "64 64"
    "128 128"
    "256 256"
    "512 512"
    "1024 1024"
    "2048 2048"
    "4096 4096"
    "8192 8192"
    "16384 16384"
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