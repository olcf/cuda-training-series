#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <cooperative_groups.h>

typedef int mytype;
const int test_dsize = 256;

const int nTPB = 256;

template <typename T>
__device__ unsigned predicate_test(T data, T testval){
  if (data == testval) return 0;
  return 1;
}

using namespace cooperative_groups;

// assume dsize is divisbile by nTPB
template <typename T>
__global__ void my_remove_if(const T * __restrict__ idata, const T remove_val, T * __restrict__ odata, unsigned * __restrict__ idxs, const unsigned dsize){

  __shared__ unsigned sidxs[nTPB];
  auto g = this_thread_block();
  auto gg = this_grid();
  unsigned tidx = g.thread_rank();
  unsigned gidx = tidx + nTPB*g.group_index().x;
  unsigned gridSize = g.size()*gridDim.x;
  // first use grid-stride loop to have each block do a prefix sum over data set
  for (unsigned i = gidx; i < dsize; i+=gridSize){
    unsigned temp = predicate_test(idata[i], remove_val);
    sidxs[tidx] = temp;
    for (int j = 1; j < g.size(); j<<=1){
      FIXME
      if (j <= tidx){ temp +=  sidxs[tidx-j];}
      FIXME
      if (j <= tidx){ sidxs[tidx] = temp;}}
    idxs[i] = temp;
    FIXME}
  // grid-wide barrier
  FIXME
  // then compute final index, and move input data to output location
  unsigned stride = 0;
  for (unsigned i = gidx; i < dsize; i+=gridSize){
    T temp = idata[i];
    if (predicate_test(temp, remove_val)){
      unsigned my_idx = idxs[i];
      for (unsigned j = 1; (j-1) < (g.group_index().x+(stride*gridDim.x)); j++) my_idx += idxs[j*nTPB-1];
      odata[my_idx-1] = temp;}
    stride++;}
}

int main(){
  // data setup
  mytype *d_idata, *d_odata,  *h_data;
  unsigned *d_idxs;
  size_t tsize = ((size_t)test_dsize)*sizeof(mytype);
  h_data = (mytype *)malloc(tsize);
  cudaMalloc(&d_idata, tsize);
  cudaMalloc(&d_odata, tsize);
  cudaMemset(d_odata, 0, tsize);
  cudaMalloc(&d_idxs, test_dsize*sizeof(unsigned));
  // check for support and device configuration
  // and calculate maximum grid size
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {printf("cuda error: %s\n", cudaGetErrorString(err)); return 0;}
  if (prop.cooperativeLaunch == 0) {printf("cooperative launch not supported\n"); return 0;}
  int numSM = prop.multiProcessorCount;
  printf("number of SMs = %d\n", numSM);
  int numBlkPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlkPerSM, my_remove_if<mytype>, nTPB, 0);
  printf("number of blocks per SM = %d\n", numBlkPerSM);
  // test 1: no remove values
  for (int i = 0; i < test_dsize; i++) h_data[i] = i;
  cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);
  cudaStream_t str;
  cudaStreamCreate(&str);
  mytype remove_val = -1;
  unsigned ds = test_dsize;
  void *args[] = {(void *)&d_idata, (void *)&remove_val, (void *)&d_odata, (void *)&d_idxs, (void *)&ds};
  dim3 grid(numBlkPerSM*numSM);
  dim3 block(nTPB);
  cudaLaunchCooperativeKernel((void *)my_remove_if<mytype>, FIXME);
  err = cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {printf("cuda error: %s\n", cudaGetErrorString(err)); return 0;}
  //validate
  for (int i = 0; i < test_dsize; i++) if (h_data[i] != i){printf("mismatch 1 at %d, was: %d, should be: %d\n", i, h_data[i], i); return 1;}
  // test 2: with remove values
  int val = 0;
  for (int i = 0; i < test_dsize; i++){
   if ((rand()/(float)RAND_MAX) > 0.5) h_data[i] = val++;
   else h_data[i] = -1;}
  thrust::device_vector<mytype> t_data(h_data, h_data+test_dsize);
  cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaLaunchCooperativeKernel((void *)my_remove_if<mytype>, FIXME);
  cudaEventRecord(stop);
  float et;
  cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&et, start, stop);
  //validate
  for (int i = 0; i < val; i++) if (h_data[i] != i){printf("mismatch 2 at %d, was: %d, should be: %d\n", i, h_data[i], i); return 1;}
  printf("kernel time: %fms\n", et);
  cudaEventRecord(start);
  thrust::remove(t_data.begin(), t_data.end(), -1);
  cudaEventRecord(stop);
  thrust::host_vector<mytype> th_data = t_data;
  // validate
  for (int i = 0; i < val; i++) if (h_data[i] != th_data[i]){printf("mismatch 3 at %d, was: %d, should be: %d\n", i, th_data[i], h_data[i]); return 1;}
  cudaEventElapsedTime(&et, start, stop);
  printf("thrust time: %fms\n", et);
  return 0;
}

