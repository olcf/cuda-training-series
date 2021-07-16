## **1. Streams Review**

For your first task, you are given a code that performs a silly computation element-wise on a vector. We already implemented a chunked version of this code using multiple CUDA streams in Homework 7. Let's start by reviewing the performance impact that CUDA streams had on this code.

Compile it using the following:

```
module load cuda/11.4.0
nvcc -o streams streams.cu -DUSE_STREAMS
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./streams
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./streams
```

To build your code on NERSC's Cori-GPU

```
module load cgpu cuda/11.4.0
nvcc -o streams streams.cu -DUSE_STREAMS
```

To run during the node reservation (10:30-12:30 Pacific time on July 16):
```
module load cgpu cuda/11.4.0
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8 ./streams
```

or grab a GPU node first, then run interactively:
```
module load cgpu cuda 
salloc -C gpu -N 1 -t 60 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8
srun -n 1 ./streams
```

To run outside of the node reservation window:
Same steps as above, but do not include "--reservation=cuda_training -q shared" in the srun or salloc commands.

In this case, the output will show the elapsed time of the non-overlapped version of the code compared to the overlapped version of the code. The non-overlapped version of the code copies the entire vector to the device, then launches the processing kernel, then copies the entire vector back to the host. In the overlapped version, the vector is broken up into chunks, and then each chunk is copied and processed asynchronously on the GPU using CUDA streams.

You can also run this code with Nsight Systems if you wish to observe the overlapping behavior:

On Summit:
```
module load nsight-systems
lsfrun nsys profile -o <destination_dir>/streams.qdrep ./streams
```

On Cori:
```
module load nsight-systems
srun -n 1 nsys profile -o <destination_dir>/streams.qdrep ./streams
```

Note that you will have to copy this file over to your local machine and install Nsight Systems for visualization. You can download Nsight Systems here:
https://developer.nvidia.com/nsight-systems

This visual output should show you the sequence of operations (*cudaMemcpy* Host to Device, kernel call, and *cudaMemcpy* Device To Host).
When you run the code, there will be a verification check performed, to make sure you have processed the entire vector correctly, in chunks. If you pass the verification test, the program will display the elapsed time of the streamed version. The overlapped version of the code should be about 2X faster (i.e. half the duration) of the non-streamed version. If you profiled the code using Nsight Systems, you should be able to confirm that there is indeed overlap of operations by zooming in on the portion of execution related to kernel launches. You can see the non-overlapped version run, followed by the overlapped version. Not only should the overlapped version be faster, you should see an interleaving of computation and data transfer operations.

## **2. OpenMP + CUDA Streams**

For this particular application, launching kernels asynchronously from a single CPU thread is sufficient. However, for legacy HPC applications that use OpenMP for on-node shared memory processing, that may not be the case. Many of these applications utilize MPI for distributing work across nodes, and they use OpenMP for better on-node shared memory processing. However, each OpenMP thread may still have quite a bit of work that can benefit from GPU acceleration, albeit not enough work to saturate the GPU on its own. In cases like this, we can combine OpenMP threads with CUDA streams to make sure our GPU is fully utilized.

In order to simulate this behavior, your task is to distribute the processing of this code's vector chunks across OpenMP threads. If done correctly, each thread will submit work to the GPU asynchronously using the CUDA streams decomposition that is already present in the code. Note that this will have no performance impact on this particular sample code. The objective is to show that we can combine CPU thread parallelism with CUDA streams in order to achieve concurrent execution on one or more GPUs.

Once you have inserted your OpenMP statement(s), compile and run using the following instructions.

On Summit:
```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
jsrun -n1 -a1 -c8 -bpacked:8 -g1 ./streams
```

On Cori:
```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 1 -c 8 ./streams
```

What does the performance look like compared to exercise 1? It should look pretty similar. How about when you profile the code? Unfortunately, the profiler currently requires some serialization when profiling across CPU threads, so you should actually see slower performance compared to the non-overlapped version. This should be reflected in the resulting qdrep file. Notice that we don't observe nearly as much concurrent execution on the GPU. This is something we are working on, and future versions of the profiler suffer from this limitation.

If you need help, refer to *streams_solution.cu*.

## **3. Bonus Task - Multi-GPU**

Remember that a CUDA stream is tied to a particular GPU. How can we combine CPU threading with more than a single GPU? If you're feeling adventurous, try adapting this homework's code to submit work to 4 GPUs, instead of just one. Note that this will require keeping track of which CUDA stream was bound to which GPU when it was created. Feel free to increase the problem size in order to ensure that there is enough work to observe a performance impact. Compile and run your code using the following instructions. 

On Summit:
```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
jsrun -n1 -a1 -c8 -bpacked:8 -g4 ./streams
```

On Cori:
```
nvcc -Xcompiler -fopenmp -o streams streams.cu -DUSE_STREAMS
export OMP_NUM_THREADS=8
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_training -q shared -G 4 -c 8 ./streams
```
