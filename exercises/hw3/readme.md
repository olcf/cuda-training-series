## **1. Vector Add**

We'll use a slight variation on the vector add code presented in a previous homework (*vector_add.cu*).  Edit the code to build a complete vector_add program. You can refer to *vector_add_solution.cu* for a complete example.  For this example, we have made a change to the kernel to use something called a grid-stride loop.  This topic will be dealt with in more detail in a later training session, but for now we can describe it as a flexible kernel design method that allows a simple kernel to handle an arbitrary size data set with an arbitrary size "grid", i.e. the configuration of blocks and threads associated with the kernel launch.  If you'd like to read more about grid-stride loops right now, you can visit https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

As we will see, this flexibility is important for our investigations in section 2 of this homework session.  However, as before, all you need to focus on are the FIXME items, and these sections will be identical to the work you did in a previous homework assignment.  If you get stuck, you can refer to the solution *vector_add_solution.cu*.

Note that this skeleton code includes something we didn't cover in lesson 1: CUDA error checking.  Every CUDA runtime API call returns an error code.  It's good practice (especially if you're having trouble) to rigorously check these error codes.  A macro is given that will make this job easier.  Note the special error checking method after a kernel call.

After editing the code, compile it using the following:

```
module load cuda
nvcc -o vector_add vector_add.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./vector_add
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./vector_add
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./vector_add
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./vector_add
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.


We've also changed the problem size from the previous example, so correct output should look like this:

```
A[0] = 0.120663
B[0] = 0.615704
C[0] = 0.736367
```

the actual numerical values aren't too important, as long as C[0] = A[0] + B[0]

## **2. Profiling Experiments**

Our objective now will be to explore some of the concepts we learned in the lesson.  In particular we want to see what effect grid sizing (choice of blocks, and threads per block) have on performance.  We could do analysis like this using host-code-based timing methods, but we'll introduce a new concept, using a GPU profiler.  In a future session, you'll learn more about the GPU profilers (Nsight Compute and Nsight Systems), but for now we will use Nsight Compute in a fairly simple fashion to get some basic data about kernel behavior, to use for comparison.
(If you'd like to read more about the Nsight profilers, you can start here: https://devblogs.nvidia.com/migrating-nvidia-nsight-tools-nvvp-nvprof/)

First, note that the code has these two lines in it:

```
  int blocks = 1;  // modify this line for experimentation
  int threads = 1; // modify this line for experimentation
```

These lines control the grid sizing.  The first variable blocks chooses the total number of blocks to launch.  The second variable threads chooses the number of threads per block to launch.  This second variable must be constrained to choices between 1 and 1024, inclusive.  These are limits imposed by the GPU hardware.

Let's consider 3 cases.  In each case, we will modify the blocks and threads variables, recompile the code, and then run the code under the Nsight Compute profiler.

Nsight Compute is installed as part of newer CUDA toolkits (10.1 and newer), but the path to the command line tool may or may not be set up as part of your CUDA install.  Therefore it may  be necessary to specify the complete command line to access the tool.  We will demonstrate that here with our invocations.

For the following profiler experiments, we will assume you have loaded the profile module and acquired a node for interactive usage:

```
module load nsight-compute
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is /bin/bash
```

### **2a.  1 block of 1 thread**

For this experiment, leave the code as you have created it to complete exercise 1 above.  When running the code you may have noticed it takes a few seconds to run, however the duration is not particularly long.  This raises the question "how much of that time is the kernel running?"  The profiler can help us answer that question, and we can use this duration (or various other characteristics) as indicators of "performance" for comparison.  The kernel is designed to do the same set of arithmetic calculations regardless of the grid sizing choices, so we can say that shorter kernel duration corresponds to higher performance.

If you'd like to get a basic idea of "typical" profiler output, you could use the following command:

```
nv-nsight-cu-cli ./vector_add
```

However for this 1 block/1 thread test case, the profiler will spend several minutes assembling the requested set of information.  Since our focus is on kernel duration, we can use a command that allows the profiler to run more quickly:

```
nv-nsight-cu-cli  --section SpeedOfLight --section MemoryWorkloadAnalysis ./vector_add
```

This will allow the profiler to complete its work in under a minute.

We won't parse all the output, but we're interested in these lines:

```
Duration                                                                        second                           2.86
```

and:

```
Memory Throughput                                                         Mbyte/second                         204.25
```

The above indicate that our kernel took about 3 seconds to run and achieved around 200MB/s "throughput" i.e. combined read and write activity, to the GPU memory.  A Tesla V100 has around 700-900 GB/s of available memory throughput, so this code isn't using the available memory bandwidth very well, amongst other issues.  Can we improve the situation with some changes to our grid sizing?

### **2b.  1 block of 1024 threads**

In our training session, we learned that we want "lots of threads".  More specifically we learned that we'd like to deposit as many as 2048 threads on a single SM, and ideally we'd like to do this across all the SMs in the GPU.  This allows the GPU to do "latency hiding" which we said was very important for GPU performance, and in the case of this code, the extra thread behavior will help with memory utilization, as well, as we shall see.  In fact, for this code, "lots of threads/latency hiding" and "efficient use of memory" are two sides of the same coin.  This will become more evident in the next training session.

So let's take a baby step with our code.  Let's change from 1 block of 1 thread to 1 block of 1024 threads. As we've learned, this structure isn't very good, because it can use at most a single SM on our GPU, but can it improve performance at all?

Edit the code to make the changes to the threads (1024) variable only.  Leave the blocks variable at 1. Recompile the code and then rerun the same profiler command.  What are the kernel duration and (achieved) memory throughput now?

(You should now observe a kernel duration that drops from the second range to the millisecond range, and the memory throughput should now be in the GB/s instead of MB/s)

### **2c. 160 blocks of 1024 threads**

Let's fill the GPU now.  We learned that a Tesla V100 has 80 SMs, and each SM can handle at most 2048 threads.  If we create a grid of 160 blocks, each of 1024 threads, this should allow for maximum "occupancy" of our kernel/grid on the GPU.  Make the necessary changes to the blocks (= 160) variable (the threads variable should already be at 1024 from step 2b), recompile the code, and rerun the profiler command as given in 2a.  What is the performance (kernel duration) and achieved memory throughput now?

(You should now observe a kernel duration that has dropped to the microsecond range - ~500us  - and a memory throughput that should be "close" to the peak theoretical of 900GB/s for a Tesla V100).

For the Tesla V100 GPU, this calculation of 80 SMs * 2048 threads/SM = 164K threads is our definition of "lots of threads". 
