# **Task 1**

In this task we will explore using compute-sanitizer.  A complete tiled matrix-multiply example code is provided in the CUDA programming guide. The *task1.cu* code includes this code with a few changes, and also a main() routine to drive the operation.  You are providing support services to a cluster user community, and one of your users has presented this code with the report that "CUDA error checking doesn't show any errors, but I'm not getting the right answer.  Please help!"

First, compile the code as follows, and run the code to observe the reported behavior:

```
module load cuda
nvcc -arch=sm_70 task1.cu -o task1 -lineinfo
```

We are compiling the code for the GPU architecture being used (Volta SM 7.0 in this case) and we are also compiling with --lineinfo switch. You know as a CUDA support engineer that this will be a useful switch when it comes to using compute-sanitizer.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./task1
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./task1
```

To build your code on NERSC's Cori-GPU

```
module load cgpu cuda/11.4.0
nvcc -arch=sm_70 task1.cu -o task1 -lineinfo
```

To run during the node reservation (10:30-12:30 Pacific time on September 14):
```
module load cgpu cuda/11.4.0
srun -C gpu -N 1 -n 1 -t 10 -A ntrain --reservation=cuda_debug -q shared -G 1 -c 1 ./task1
```

or grab a GPU node first, then run interactively:
```
module load cgpu cuda 
salloc -C gpu -N 1 -t 60 -A ntrain --reservation=cuda_debug -q shared -G 1 -c 1
srun -n 1 ./task1
```

To run outside of the node reservation window:
Same steps as above, but do not include "*--reservation=cuda_debug -q shared*" in the srun or salloc commands.

If this code produces the correct matrix result, it will display:

```
Success!
```

But unfortunately we don't see that.

## Part A 

Use basic *compute-sanitizer* functionality (no additional switches) to identify a problem in the code. Using the output from *compute-sanitizer*, identify the offending line of code. Fix this issue.

Hints:
  - Remember that *-lineinfo* will cause compute-sanitizer (in this usage) to report the actual line of code that is causing the problem
  - Even if you didn't have this information (line number) could you use other compute sanitizer information to quickly deduce the line to focus on in this case?  You could use the type of memory access violation as a clue.  Which lines of code in the kernel are doing that type of memory access (hint, there is only one line of kernel code that is doing this.)
  - Memory access problems are often caused by indexing errors.  See if you can spot an indexing error that may lead to this issue (hint - the classic computer science "off by one" error.)
  - Refer to *task1_solution.cu* if you get stuck

## Part B

Yay! You sorted out the problem, made the change to indexing, and now the code prints "Success!"  It's time to send the user on their way. Or is it? Could there be other errors?  Use additional compute-sanitizer switches (*--tool racecheck*, *--tool initcheck*, *--tool synccheck*) to identify other "latent" issues. Fix them.

Hints:
  - The only tool that should report a problem at this point is the racecheck tool.
  - See if you can use the line number information embedded in the error reports to identify the trouble "zone" in the kernel code
  - Since you know that the racecheck tool reports race issues with shared memory usage (only), and that these often involve missing synchronization, can you identify the right place to insert appropriate synchronization into the kernel code? Try experimenting. Inserting additional synchronization into a CUDA kernel code usually does not break code correctness.
  - Refer to *task1_solution.cu* if you get stuck

# **Task 2**

In this task we will explore basic usage of cuda-gdb. Once again you are providing user support at a cluster help desk. The user has a code that produces a *-inf* (negative floating-point infinity) result, and that is not expected. The code consists of a transformation operation (one data element created/modified per thread) followed by a reduction operation (per-thread results summed together). The output of the reduction is *-inf*. See if you can use *cuda-gdb* to identify the problem and rectify it.

To prepare to use *cuda-gdb*, its necessary to compile a debug project. Therefore compile the code as follows:

```
nvcc -arch=sm_70 task2.cu -o task2 -G -g -std=c++14
```

You can then start debugging.

On Summit:

```
jsrun -n1 -a1 -c1 -g1 cuda-gdb ./task2
```

On Cori:

```
srun -n 1 ./task2
```

Don't forget that you cannot inspect device data until you are stopped after a device-code breakpoint.

Once you have identified the source of the issue, see if you can propose a simple code modification to work around the issue. If you get stuck on this part (proposing a solution), refer to the *task2_solution.cu*. Careful code inspection will likely immediately point out the issue, however the purpose of this task is not actually to fix the code this way, but to learn to use *cuda-gdb*.

Hints:
 - The code is attempting to estimate the sum of an alternating harmonic series (ahs), whose sum should be equal to the natural log of 2.
 - The code is broken into two parts: the ahs term generator (produced by the device function ahs) which takes only the index of the term to generate, and a standard sweep parallel reduction, similar to the content in session 5 of this training series.
 - Generally speaking, floating point arithmetic on *inf* or *-inf* inputs will produce a *inf* or *-inf* output
 - Decide whether you think the *-inf* is likely to appear as a result of the initial transformation operation, or the subsequent reduction operation
 - Use this reasoning to choose a point for an initial breakpoint
 - Inspect data to see if you can observe *-inf* in any of the intermediate data
 - Use this observation to repeat the process of setting a breakpoint and inspecting data
 - Alternatively, work linearly through the code, setting an initial breakpoint and single-stepping, to see if you can observe incorrect data
 - You may need to change thread focus or observe data belonging to other threads
 - The reduction also offers the opportunity to tackle this problem via divide-and-conquer, or binary searching
 - Consider reducing the problem size (i.e. length of terms to generate the estimate) to simplify your debug effort
