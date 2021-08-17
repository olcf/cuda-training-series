#ifndef NO_MPI
#include <mpi.h>
#endif
#include <cstdio>
#include <chrono>
#include <iostream>

__global__ void kernel (double* x, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        x[i] = 2 * x[i];
    }
}

int main(int argc, char** argv) {
#ifndef NO_MPI
    int rank, num_ranks;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Total problem size
    size_t N = 1024 * 1024 * 1024;

    if (argc >= 2) {
        N = atoi(argv[1]);
    }

#ifdef NO_MPI
    // If not using MPI, specify at command line how many "ranks" there are
    int num_ranks = 1;
    if (argc >= 3) {
        num_ranks = atoi(argv[2]);
    }
#endif

    // Problem size per rank (assumes divisibility of N)
    size_t N_per_rank = N / num_ranks;

    double* x;
    cudaMalloc((void**) &x, N_per_rank * sizeof(double));

    // Number of repetitions

    const int num_reps = 1000;

    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    int threads_per_block = 256;
    size_t blocks = (N_per_rank + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < num_reps; ++i) {
        kernel<<<blocks, threads_per_block>>>(x, N_per_rank);
        cudaDeviceSynchronize();
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Time per kernel = " << duration.count() / (double) num_reps << " ms " << std::endl;

#ifndef NO_MPI
    MPI_Finalize();
#endif
}
