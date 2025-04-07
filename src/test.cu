#include <iostream>
#include <vector>
#include <cmath>
#include <cufft.h>

// Define the dimensions of the 4D array
#define NX 32
#define NY 32
#define NZ 32
#define NW 32

int main() {
  // 1. Allocate host memory
  std::vector<cuComplex> h_data(NX * NY * NZ * NW);

  // Initialize host data (example: complex exponentials)
  for (int w = 0; w < NW; ++w) {
    for (int z = 0; z < NZ; ++z) {
      for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
          float angle = 2.0f * M_PI * (x * 1.0f / NX + y * 2.0f / NY + z * 3.0f / NZ + w * 4.0f / NW);
          h_data[w * NZ * NY * NX + z * NY * NX + y * NX + x] = cuComplex{cosf(angle), sinf(angle)};
        }
      }
    }
  }

  // 2. Allocate device memory
  cuComplex *d_data;
  cudaMalloc((void**)&d_data, NX * NY * NZ * NW * sizeof(cuComplex));

  // 3. Copy data from host to device
  cudaMemcpy(d_data, h_data.data(), NX * NY * NZ * NW * sizeof(cuComplex), cudaMemcpyHostToDevice);

  // 4. Create CUFFT plan for 4D FFT
  cufftHandle plan4D;
  int n[4] = {NX, NY, NZ, NW};
  cufftPlanMany(
      &plan4D,
      4,      // rank
      n,      // n (dimensions)
      nullptr, // inembed (if stride is used)
      1,      // istride
      0,      // idist
      nullptr, // onembed (if stride is used)
      1,      // ostride
      0,      // odist
      CUFFT_C2C, // type
      1       // batch
  );

  // 5. Execute the 4D FFT
  cufftExecC2C(plan4D, d_data, d_data, CUFFT_FORWARD);

  // 6. Copy result back to host (optional)
  std::vector<cuComplex> h_result(NX * NY * NZ * NW);
  cudaMemcpy(h_result.data(), d_data, NX * NY * NZ * NW * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  // 7. Clean up
  cufftDestroy(plan4D);
  cudaFree(d_data);

  std::cout << "4D FFT completed successfully." << std::endl;

  // You can now process or verify the results in h_result

  return 0;
}