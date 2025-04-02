#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

__global__ void hello_cuda() {
    printf("Hello from CUDA device... \n");
}

int main(int argc, char** argv) {
    hello_cuda<<<1,1>>>();
    printf("Hello from CPU world... \n");
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}