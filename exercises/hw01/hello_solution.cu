#include <stdio.h>

__global__ void hello(){

  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){
  hello<<<2,2>>>();
  cudaDeviceSynchronize();
}

