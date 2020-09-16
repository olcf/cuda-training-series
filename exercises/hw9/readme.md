# **1. Exploring Threadblock-Level Groups**

## **1a. Creating Groups**

First, you should take the *task1.cu* code, and complete the sections indicated by **FIXME** to provide a proper thread-block group, and assign that group to the group being used for printout purposes.  You should only need to modify the 2 lines containing **FIXME** for this first step.

You can compile your code as follows:

```bash
module load cuda
nvcc -arch=sm_70 -o task1 task1.cu -std=c++11
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++. Note that because we're using C++11 (which is required for cooperative groups) we need a sufficiently modern compiler (gcc >= 5 should be sufficient). If you're on Summit, make sure to do `module load gcc` because the system default gcc is not recent enough.

To run your code, we will use an LSF command:

```bash
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./task1
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```bash
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./task1
```

To run your code at NERSC on Cori, we can use Slurm:

```bash
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./task1
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```bash
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./task1
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

Correct output should look like this:

```bash
group partial sum: 256
```

If you need help, refer to the *task1_solution1.cu* file. (which contains the solution for tasks 1a, 1b, and 1c)

## **1b. Partitioning Groups**

Next uncomment the next line that starts with the auto keyword, and complete that line to use the previously created thread block group and subdivide it into a set of 32-thread partitions, using the dynamic (runtime) partitioning method.

Compile and run the code as above.  correct output should look like:

```bash
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
group partial sum: 32
```

## **1c. Third Group Creation/Decomposition**

Now perform the 3rd group creation/decomposition.

Compile and run the code as above.  Correct output should look like:

```bash
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
group partial sum: 16
```

# **2. Exploring Grid-Wide Sync**

One of the motivations suggested for a grid-wide sync is to combine algorithm phases which need to be completed in sequence, and would normally be realized with separate CUDA kernel calls.  In this case, the kernel launch boundary provides an implicit/effective grid-wide sync.  However cooperative groups provides the possibility of a grid wide sync directly in kernel code, rather than at a kernel launch boundary.

One such algorithm would be stream compaction.  Stream compaction is used in many places, and fundamentally seeks to reduce the length of a data stream using a particular removal heuristic or predicate test.  For example, if we had the following data stream:

```bash
3 4 3 7 0 5 0 8 0 0 0 4
```

we could do stream compaction by removing the zeroes, ending up with:

```bash
3 4 3 7 5 8 4
```

Like many reduction type algorithms (the output here is potentially much smaller than the input), we can easily imagine how to do this in a serial fashion, but a fast parallel stream compaction requires some additional thought.  A common approach is to use a prefix sum.  A prefix sum is a data set, where each data item in the set represents the sum of the previous input elements from the beginning of the input to that point.  We can use a prefix sum to help parallelize our stream compaction.  We start by creating an array of ones and zeroes, where there is a one corresponding to the element we want to keep, and zero for the element we want to discard:

```bash
3 4 3 7 0 5 0 8 0 0 0 4 (input data)
1 1 1 1 0 1 0 1 0 0 0 1 (filtering of input)
```

We then do an exclusive prefix sum on that filtered array (exclusive means only the elements "to the left" are included in the sum.  The element at that position is excluded).

```bash
3 4 3 7 0 5 0 8 0 0 0 4 (input data)
1 1 1 1 0 1 0 1 0 0 0 1 (filtering of input)
0 1 2 3 4 4 5 5 6 6 6 6 (exclusive prefix sum of filtered data)
```

This prefix sum now contains the index into the output array that the input position should be copied to.  We only copy a position from input to output if the corresponding filter element is not zero.  This demonstrates how to use a prefix sum to assist with a stream compaction, but doesn't identify how to do the prefix sum in parallel, efficiently.  A full treatment here is beyond the scope of this document, but you can refer here for a good treatise: https://people.eecs.berkeley.edu/~driscoll/cs267/papers/gpugems3_ch39.html  Some key takeaways are that a prefix sum has a sweeping operation, not unlike the sweeping operation that is successively performed in a parallel reduction, but there are key differences.  Two of these key differences are that the sweep is from "left" to "right" in the prefix sum whereas it is usually from right to left  in a typical parallel reduction, and also that the break points (i.e. the division of threads participating at each sweep phase) is different.

When parallelizing a prefix sum, we often require multiple phases, for example a thread-block level scan (prefix-sum) operation, followed by another operation to "fix up" the threadblock level results based on the data from other ("previous") thread blocks.  These phases may require a grid-wide sync, and  typical scan from a library such as thrust will use multiple kernel calls.   Let's see if we can do it in a single kernel call. You won't have to write any scan code, other than inserting appropriate cooperative group sync points.  We need sync points at the threadblock leve (based on the threadblock level group created for you) and also at the grid level.

Start with the *task2.cu* code, and perform 2 things:

- Modify the **FIXME** statements in the kernel to insert appropriate sync operations as requested, based on the two group types created at the top of the kernel.  Only one grid-wide sync point is needed, the others are all thread-block-level sync points.
- In the host code, modify the **FIXME** statements to do a proper cooperative launch.  The launch function is already provided, you just need to fill in the remaining 4 arguments.  Refer to the *task2_solution.cu* file for help, or refer to the cuda runtime API documentation for the launch function: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73  

Once you have made the above modification, compile your code as follows:

```bash
nvcc -arch=sm_70 -o task2 task2.cu -rdc=true -std=c++11
```

and run it as follows:

```bash
lsfrun ./task2
```

Correct output should be simply:

```bash
number of SMs = 80
number of blocks per SM = 8
kernel time: 0.043872ms
thrust time: 0.083200ms
```

(The above is representative, if you run it on a GPU that is different than Tesla V100, you may see different data, but should only see the above 4 lines)

The above informational data contains "*occupancy*" information produced by the occupancy API.  Note that we are able to put 8 of these 256 thread threadblocks on a single SM, for the full maximum theoretical 2048 thread complement per SM. This is 100% occupancy (the kernel is fairly simple and low in resource utilization/requirements).

The code has silent validation built in, so no actual results are printed, other than the above informational data.  If you got a "*mismatch*" message, something is wrong with your implementation.

Optional:

This task2 code compares the operation to an equivalent operation in thrust.  For this trivially small data set size, our monolithic kernel seems to be faster than thrust.  Run this small data set size using the nsight-compute profiler to confirm for yourself that thrust is actually doing 2 kernel calls to solve this problem:

```bash
module load nsight-compute
lsfrun nv-nsight-cu-cli ./task2
```

Now make the data set larger.  A reasonable upper limit might be 32M elements.  Make sure to chose a number that is divisble by 256, the threadblock size. For example, change:

```cpp
const int test_dsize = 256;
```

to something like:

```cpp
const int test_dsize = 1048576*16;
```

and recompile and rerun the code.  Now which is faster, thrust or our naive code?

Takeaway: don't write your own code if you can find a high-quality library implementation.  This is especially true for more complex algorithms like sorting, prefix sums, and matrix multiply.
