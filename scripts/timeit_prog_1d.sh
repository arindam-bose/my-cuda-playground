#!/bin/bash

# Create the executable if the arguments are correct
if [ -z "$1" ]; then
    echo "This script should be called with atleast one argument:" >&2
    echo "e.g. $0 cufft" >&2
    echo "     $0 fftw" >&2
    exit 2
elif [ "$1" == "cufft" ]; then
    echo "Generating binary for cufft"
    PROGRAM=cufft1d
    ARCH="DUMMY"
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "fftw" ]; then
    echo "Generating binary for fftw"
    PROGRAM=fftw1d
    ARCH=$(uname -m)
    gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm
else
    echo "First argument should be {cufft|fftw}"
    exit 2
fi

# Number of iterations is optional second argument, default 5
NITER=${2:-"5"}

# Output file
if [ $ARCH == "DUMMY" ]; then
    OUTPUT_FILE="build/time_results_${PROGRAM}.txt"
else
    OUTPUT_FILE="build/time_results_${PROGRAM}_${ARCH}.txt"
fi

# First line is the program name
echo -e "$PROGRAM" > "$OUTPUT_FILE"

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
    echo "--------------------------------------------"
    echo "Running: $EXECUTABLE $ARGS $NITER"
    result=$($EXECUTABLE $ARGS $NITER)
    echo "Elapsed time: $result s"
    echo -e "$ARGS" "$result" >> "$OUTPUT_FILE"
done

echo "Outputs are saved at: $OUTPUT_FILE"