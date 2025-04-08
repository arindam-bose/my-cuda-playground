#!/bin/bash

# Create the executable if the arguments are correct
if [ -z "$1" ]; then
    echo "This script should be called with atleast one argument:" >&2
    echo "e.g. $0 cufft4" >&2
    echo "     $0 cufft31" >&2
    echo "     $0 cufft22" >&2
    echo "     $0 fftw" >&2
    exit 2
elif [ "$1" == "cufft4" ]; then
    echo "Generating binary for cufft4"
    PROGRAM=cufft4d_4x1d
    ARCH="DUMMY"
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "cufft31" ]; then
    echo "Generating binary for cufft31"
    PROGRAM=cufft4d_3d1d
    ARCH="DUMMY"
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "cufft22" ]; then
    echo "Generating binary for cufft22"
    PROGRAM=cufft4d_2d2d
    ARCH="DUMMY"
    nvcc src/$PROGRAM.cu -o build/$PROGRAM --ptxas-options=-v --use_fast_math -lcufft
elif [ "$1" == "fftw" ]; then
    echo "Generating binary for fftw"
    PROGRAM=fftw4d
    ARCH=$(uname -m)
    gcc src/$PROGRAM.c -o build/$PROGRAM -lfftw3 -lm
else
    echo "First argument should be {cufft4|cufft31|cufft22|fftw}"
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
    "8x8x8x8"
    "16x16x16x16"
    "32x32x32x32"
    "64x64x64x64"
    "64x64x512x128"
    "64x64x512x1024"
    "128x128x128x128"
)

# Define the executable name
EXECUTABLE="./build/$PROGRAM"

# Loop through each set of arguments and time the execution
for ARGS in "${ARGS_LIST[@]}"
do
    echo "--------------------------------------------"
    # Replace the x with space to be used as proper arguments
    ARGS_R=$(echo "$ARGS" | tr x " ")
    echo "Running: $EXECUTABLE $ARGS_R $NITER"
    result=$($EXECUTABLE $ARGS_R $NITER)
    echo "Elapsed time: $result s"
    echo -e "$ARGS" "$result" >> "$OUTPUT_FILE"
done

echo "Outputs are saved at: $OUTPUT_FILE"