#!/bin/bash

# Create the executable if the arguments are correct
if [ -z "$1" ]; then
    echo "This script should be called with atleast one argument:" >&2
    echo "e.g. $0 cufft4" >&2
    echo "     $0 cufft31" >&2
    echo "     $0 fftw" >&2
    exit 2
elif [ "$1" == "cufft4" ]; then
    echo "Generating binary for cufft4"
    PROGRAM=cufft4d_4x1d
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "cufft31" ]; then
    echo "Generating binary for cufft31"
    PROGRAM=cufft4d_3d1d
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "fftw" ]; then
    echo "Generating binary for fftw"
    PROGRAM=fftw4d
    gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm
else
    echo "First argument should be {cufft4|cufft31|fftw}"
    exit 2
fi

OUTPUT_FILE="build/time_results_$PROGRAM.txt"

# First line is the program name
echo -e "$PROGRAM" > "$OUTPUT_FILE"

# Define argument sets (each line represents a set of arguments)
ARGS_LIST=(
    "8 8 8 8"
    "16 16 16 16"
    "32 32 32 32"
    "64 64 64 64"
    "64 64 512 128"
    "64 64 512 1024"
    "128 128 128 128"
)

# Define the executable name
EXECUTABLE="./build/$PROGRAM"

# Loop through each set of arguments and time the execution
for ARGS in "${ARGS_LIST[@]}"
do
    echo "Running: $EXECUTABLE $ARGS"
    result=$($EXECUTABLE $ARGS)
    echo "Elapsed time: $result s"
    echo -e "$ARGS" "$result" >> "$OUTPUT_FILE"
    echo "---------------------------------------"
done