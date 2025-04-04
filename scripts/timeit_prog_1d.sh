#!/bin/bash

# Create the executable if the arguments are correct
if [ -z "$1" ]; then
    echo "This script should be called with atleast one argument:" >&2
    echo "e.g. $0 nvcc" >&2
    echo "     $0 gcc" >&2
    exit 2
elif [ "$1" == "nvcc" ]; then
    echo "Generating binary for nvcc"
    PROGRAM=cufft1d
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "gcc" ]; then
    echo "Generating binary for gcc"
    PROGRAM=fftw1d
    gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm
else
    echo "First argument should be {nvcc|gcc}"
fi

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