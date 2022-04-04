#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

// modifiable
typedef float ft;
const int chunks = 64;
const size_t ds = 1024*1024*chunks;
const int count = 22;
const int num_gpus = 4;

// not modifiable
const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;
__device__ float gpdf(float val, float sigma) {
  return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

__device__ double gpdf(double val, double sigma) {
  return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}

// compute average gaussian pdf value over a window around each point
__global__ void gaussian_pdf(const ft * __restrict__ x, ft * __restrict__ y, const ft mean, const ft sigma, const int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    ft in = x[idx] - (count / 2) * 0.01f;
    ft out = 0;
    for (int i = 0; i < count; i++) {
      ft temp = (in - mean) / sigma;
      out += gpdf(temp, sigma);
      in += 0.01f;
    }
    y[idx] = out / count;
  }
}

// error check macro
#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

// host-based timing
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int main() {
  ft *h_x, *d_x[num_gpus], *d_y[num_gpus];
  h_x = (ft *)malloc(ds * sizeof(ft));

  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_x[i], ds * sizeof(ft));
    cudaMalloc(&d_y[i], ds * sizeof(ft));
  }
  cudaCheckErrors("allocation error");

  for (int i = 0; i < num_gpus; i++) {
    for (size_t j = 0; j < ds; j++) {
      h_x[j] = rand() / (ft)RAND_MAX;
    }
    cudaSetDevice(i);
    cudaMemcpy(d_x[i], h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
  }
  cudaCheckErrors("copy error");

  unsigned long long et1 = dtime_usec(0);

  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    gaussian_pdf<<<(ds+255)/256, 256>>>(d_x[i], d_y[i], 0.0, 1.0, ds);
  }
  cudaDeviceSynchronize();
  cudaCheckErrors("execution error");

  et1 = dtime_usec(et1);
  std::cout << "elapsed time: " << et1/(float)USECPSEC << std::endl;

  return 0;
}
