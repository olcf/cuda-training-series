## **1. Investigating Copy-Compute Overlap**

For your first task, you are given a code that performs a silly computation element-wise on a vector. You can initially compile, run and profile the code if you wish. 

compile it using the following:

```
module load cuda
nvcc -o overlap overlap.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./overlap
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./overlap
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 -G 1 -c 10 ./overlap
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 -G 1 -c 10
srun -n 1 ./overlap
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

In this case, the output will show the elapsed time of the non-overlapped version of the code. This code copies the entire vector to the device, then launches the processing kernel, then copies the entire vector back to the host.

You can also run this code with Nsight Systems if you wish:

```
module load nsight-systems
lsfrun nsys profile -o <destination_dir>/overlap.qdrep ./overlap
```

Note that you will have to copy this file over to your local machine and install Nsight Systems for visualization. You can download Nsight Systems here:
https://developer.nvidia.com/nsight-systems

This visual output should show you the sequence of operations (*cudaMemcpy* Host to Device, kernel call, and *cudaMemcpy* Device To Host). Note that there is an initial "warm-up" run of the kernel; disregard this. You should be able to witness that the start and duration of each operating is indicating that there is no overlap.

Your objective is to create a fully overlapped version of the code. Use your knowledge of streams to create a version of the code that will issue the work in chunks, and for each chunk perform the copy to device, kernel launch, and copy to host in a single stream, then modifying the stream for the next chunk. The work has been started for you in the section of code after the #ifdef statement. Look for the FIXME tokens there, and replace each FIXME with appropriate code to complete this task.

When you have something ready to test, compile with this additional switch:

```
nvcc -o overlap overlap.cu -DUSE_STREAMS
```

If you run the code, there will be a verification check performed, to make sure you have processed the entire vector correctly, in chunks. If you pass the verification test, the program will display the elapsed time of the streamed version.  You should be able to get to at least 2X faster (i.e. half the duration) of the non-streamed version. If you wish, you can also run this code with the Nsight Systems profiler using the above given command. This will generate a visual output, and you should be able to confirm that there is indeed overlap of operations by zooming in on the portion of execution related to kernel launches. You can see the non-overlapped version run, followed by the overlapped version. Not only should the overlapped version be faster, you should see an interleaving of computation and data transfer operations.

If you need help, refer to *overlap_solution.cu*.

## **2. Simple Multi-GPU**

In this exercise, you are given a very simple code that performs 4 kernel calls in sequence on a single GPU. You're welcome to compile and run the code as-is. It will display an overall duration for the time taken to complete the 4 kernel calls. Your task is to modify this code to run each kernel on a separate GPU (each node on Summit actually has 6 GPUs). After completion, confirm that the execution time is substantially reduced.

You can compile the code with:

```
nvcc -o multi multi.cu
```

You can run the code on Summit with:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g4'
lsfrun ./multi
```

On Cori, make sure that you ask for an allocation with 4 GPUs, e.g.

```
srun -C gpu -N 1 -n 1 -t 10 -A m3502 -G 4 -c 40 ./multi
```

**HINT**: This exercise might be simpler than you think. You won't need to do anything with streams at all for this. You'll only need to make a simple modification to each of the for-loops.

If you need help, refer to *multi_solution.cu*.
