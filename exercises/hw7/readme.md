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

In this case, the output will show the elapsed time of the non-overlapped version of the code. This code copies the entire vector to the device, then launches the processing kernel, then copies the entire vector back to the host.

You can also run this code with nvprof if you wish:

```
lsfrun nvprof --print-gpu-trace ./overlap
```

This nvprof output should show you the sequence of operations (*cudaMemcpy* Host to Device, kernel call, and *cudaMemcpy* Device To Host). Note that there is an initial "warm-up" run of the kernel; disregard this. You should be able to witness that the start and duration of each operating is indicating that there is no overlap.

Your objective is to create a fully overlapped version of the code. Use your knowledge of streams to create a version of the code that will issue the work in chunks, and for each chunk perform the copy to device, kernel launch, and copy to host in a single stream, then modifying the stream for the next chunk. The work has been started for you in the section of code after the #ifdef statement. Look for the FIXME tokens there, and replace each FIXME with appropriate code to complete this task.

When you have something ready to test, compile with this additional switch:

```
nvcc -o overlap overlap.cu -DUSE_STREAMS
```

If you run the code, there will be a verification check performed, to make sure you have processed the entire vector correctly, in chunks. If you pass the verification test, the program will display the elapsed time of the streamed version.  You should be able to get to at least 2X faster (i.e. half the duration) of the non-streamed version. If you wish, you can also run this code with the nvprof profiler using the above given command. This will generate a lot of output, however if you examine some of the operations at the end of the output, you should be able to confirm that there is indeed overlap of operations, based on the starting time and duration of each operation.

If you need help, refer to *overlap_solution.cu*.

## **2. Simple Multi-GPU**

In this exercise, you are given a very simple code that performs 4 kernel calls in sequence on a single GPU. You're welcome to compile and run the code as-is. It will display an overall duration for the time taken to complete the 4 kernel calls. Your task is to modify this code to run each kernel on a separate GPU (the slurm reservation should put you on a machine that has 4 GPUs available to you). After completion, confirm that the execution time is substantially reduced.

You can compile the code with:

```
nvcc -o multi multi.cu
```

You can run the code with:

```
lsfrun ./multi
```

**HINT**: This exercise might be simpler than you think. You won't need to do anything with streams at all for this. You'll only need to make a simple modification to each of the for-loops.

If you need help, refer to *multi_solution.cu*.
