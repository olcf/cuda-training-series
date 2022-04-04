## **1. Comparing Reductions**

For your first task, the code is already written for you. We will compare 3 of the reductions given during the presentation: the naive atomic-only reduction, the classical parallel reduction with atomic finish, and the warp shuffle reduction (with atomic finish).

Compile it using the following:

```
module load cuda
nvcc -o reductions reductions.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++. Let's also load the Nsight Compute module:

```
module load nsight-compute
```

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 nv-nsight-cu-cli ./reductions
```

Alternatively, you may want to create an alias for your *bsub* command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun nv-nsight-cu-cli ./reductions
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./reductions
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./reductions
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

This will run the code with the profiling in its most basic mode, which is sufficient. We want to compare kernel execution times. What do you notice about kernel execution times? Probably, you won't see much difference between the parallel reduction with atomics and the warp shuffle with atomics kernel. Can you theorize why this may be? Our objective with these will be to approach theoretical limits. The theoretical limit for a typical reduction would be determined by the memory bandwidth of the GPU. To calculate the attained memory bandwidth of this kernel, divide the total data size in bytes (use N from the code in your calculation) by the execution time (which you can get from the profiler). How does this number compare to the memory bandwidth of the GPU you are running on? (You could run bandwidthTest sample code to get a proxy/estimate).

Now edit the code to change *N* from ~8M to 163840 (=640*256)

Recompile and re-run the code with profiling. Is there a bigger percentage difference between the execution time of the reduce_a and reduce_ws kernel? Why might this be?

Bonus: edit the code to change *N* from ~8M to ~32M.  recompile and run.  What happened? Why?

## **2. Create a different reduction (besides sum)**

For this exercise, you are given a fully-functional sum-reduction code, similar to the code used for exercise 1 above, except that we will use the 2-stage reduction method without atomic finish. If you wish you can compile and run it as-is to see how it works. Your task is to modify it (*only the kernel*) so that it creates a proper max-finding reduction. That means that the kernel should report the maximum value in the data set, rather than the sum of the data set. You are expected to use a similar parallel-sweep-reduction technique. If you need help, refer to the solution.

```
nvcc -o max_reduction max_reduction.cu
lsfrun ./max_reduction
```

## **3. Revisit row_sums from hw4**

For this exercise, start with the *matrix_sums.cu* code from hw4. As you may recall, the *row_sums* kernel was reading the same data set as the *column_sums* kernel, but running noticeably slower. We now have some ideas how to fix it. See if you can implement a reduction-per-row, to allow the row-sum kernel to approach the performance of the column sum kernel. There are probably several ways to tackle this problem. To see one approach, refer to the solution.

You can start just by compiling the code as-is and running the profiler to remind yourself of the performance (discrepancy).

Compile the code and profile it using Nsight Compute:

```
nvcc -o matrix_sums matrix_sums.cu
lsfrun nv-nsight-cu-cli ./matrix_sums
```

Remember from the previous session our top 2 CUDA optimization priorities: lots of threads and efficient use of the memory subsystem. The original row_sums kernel definitely misses the mark for the memory objective. What we've learned about reductions should guide you. There are probably several ways to tackle this:

 - Write a straightforward parallel reduction, run it on a row, and use a for-loop to loop the kernel over all rows
 - Assign a warp to each row, to perform everything in one kernel call
 - Assign a threadblock to each row, to perform everything in one kernel call
 - ??

Since the (given) solution may be somewhat unusual, I'll give some hints here if needed:

 - The chosen strategy will be to assign one block per row
 - We must modify the kernel launch to launch exactly as many blocks as we have rows
 - The kernel can be adapted from the reduction kernel (atomic is not needed here) from the reduce kernel code in exercise 1 above.
 - Since we are assigning one block per row, we will cause each block to perform a block-striding loop, to traverse the row.  This is conceptually similar to a grid striding loop, except each block is striding individually, one per row.  Refresh your memory of the grid-stride loop, and see if you can work this out.
 - With the block-stride loop, you'll need to think carefully about indexing

After you have completed the work and are getting a successful result, profile the code again to see if the performance of the row_sums kernel has improved:

```
nvcc -o matrix_sums matrix_sums.cu
lsfrun nv-nsight-cu-cli ./matrix_sums
```

Your actual performance here (compared to the fairly efficient column_sums kernel) will probably depend quite a bit on the algorithm/method you choose.  See if you can theorize how the various choices may affect efficiency or optimality. If you end up with a solution where the row_sums kernel actually runs faster than the column_sums kernel, see if you can theorize why this may be. Remember the two CUDA optimization priorities, and use these to guide your thinking.
