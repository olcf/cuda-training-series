#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

// error checking macro
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

#define N 500000

// Simple short kernels
__global__
void kernel_a(float* x, float* y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

__global__
void kernel_c(float* x, float* y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += 1;
}

int main(){

cudaStream_t stream1;

cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);
cublasSetStream(cublas_handle, stream1);

// Set up host data and initialize
float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));
h_y = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_x[i] = float(i);
    h_y[i] = float(i);
}

// Print out the first 25 values of h_y
for (int i = 0; i < 25; ++i){
    printf("%2.0f ", h_y[i]);
}
printf("\n");

// Set up device data
float* d_x;
float* d_y;
float d_a = 5.0;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));
cudaCheckErrors("cudaMalloc failed");

cublasSetVector(N, sizeof(h_x[0]), h_x, 1, d_x, 1); // similar to cudaMemcpyHtoD
cublasSetVector(N, sizeof(h_y[0]), h_y, 1, d_y, 1); // similar to cudaMemcpyHtoD
cudaCheckErrors("cublasSetVector failed");

// Set up graph
cudaGraph_t graph; // main graph
cudaGraph_t libraryGraph; // sub graph for cuBLAS call
std::vector<cudaGraphNode_t> nodeDependencies;
cudaGraphNode_t kernelNode1, kernelNode2, libraryNode;

cudaKernelNodeParams kernelNode1Params {0};
cudaKernelNodeParams kernelNode2Params {0};

cudaGraphCreate(&graph, 0); // create the graph
cudaCheckErrors("cudaGraphCreate failure");

// kernel_a and kernel_c use same args
void *kernelArgs[2] = {(void *)&d_x, (void *)&d_y};

int threads = 512;
int blocks = (N + (threads - 1) / threads);

// Adding 1st node, kernel_a, as head node of graph
kernelNode1Params.func = (void *)kernel_a;
kernelNode1Params.gridDim = dim3(blocks, 1, 1);
kernelNode1Params.blockDim = dim3(threads, 1, 1);
kernelNode1Params.sharedMemBytes = 0;
kernelNode1Params.kernelParams = (void **)kernelArgs;
kernelNode1Params.extra = NULL;

cudaGraphAddKernelNode(&kernelNode1, graph, NULL,
                         0, &kernelNode1Params);
cudaCheckErrors("Adding kernelNode1 failed");

nodeDependencies.push_back(kernelNode1); // manage dependecy vector

// Adding 2nd node, libraryNode, with kernelNode1 as dependency
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
cudaCheckErrors("Stream capture begin failure");

// Library call
cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);
cudaCheckErrors("cublasSaxpy failure");

cudaStreamEndCapture(stream1, &libraryGraph);
cudaCheckErrors("Stream capture end failure");

cudaGraphAddChildGraphNode(&libraryNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), libraryGraph);
cudaCheckErrors("Adding libraryNode failed");

nodeDependencies.clear();
nodeDependencies.push_back(libraryNode); // manage dependency vector

// Adding 3rd node, kernel_c, with libraryNode as dependency
kernelNode2Params.func = (void *)kernel_c;
kernelNode2Params.gridDim = dim3(blocks, 1, 1);
kernelNode2Params.blockDim = dim3(threads, 1, 1);
kernelNode2Params.sharedMemBytes = 0;
kernelNode2Params.kernelParams = (void **)kernelArgs;
kernelNode2Params.extra = NULL;

cudaGraphAddKernelNode(&kernelNode2, graph, nodeDependencies.data(),
                         nodeDependencies.size(), &kernelNode2Params);
cudaCheckErrors("Adding kernelNode2 failed");

nodeDependencies.clear();
nodeDependencies.push_back(kernelNode2); // manage dependency vector

cudaGraphNode_t *nodes = NULL;
size_t numNodes = 0;
cudaGraphGetNodes(graph, nodes, &numNodes);
cudaCheckErrors("Graph get nodes failed");
printf("Number of the nodes in the graph = %zu\n", numNodes);

cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaCheckErrors("Graph instantiation failed");

// Launch the graph instance 100 times
for (int i = 0; i < 100; ++i){
    cudaGraphLaunch(instance, stream1);
    cudaStreamSynchronize(stream1);
}
cudaCheckErrors("Graph launch failed");
cudaDeviceSynchronize();

// Copy memory back to host
cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);
cudaCheckErrors("Finishing memcpy failed");

cudaDeviceSynchronize();

// Print out the first 25 values of h_y
for (int i = 0; i < 25; ++i){
    printf("%2.0f ", h_y[i]);
}
printf("\n");

return 0;

}
