# Homework 6

These exercises will have you use Unified Memory to utilize GPUs on non-trivial data structures.

## **1. Porting Linked Lists to GPUs**

For your first task, you are given a code that assembles a linked list on the CPU, and then attempts to print an element from the list. Your task is to modify the code using UM techniques, so that the linked list can be correctly traversed either from CPU code or from GPU code. Hint: there is only one line in the file that needs to be modified to do this exercise.

Compile it using the following:

```
module load cuda
nvcc -o linked_list linked_list.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./linked_list
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./linked_list
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./linked_list
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./linked_list
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

Correct output should look like this:

```
key = 3
key = 3
```

If you need help, refer to *linked_list_solution.cu*


## **2. Array Increment**

In this exercise, you are given a code that increments a large array on the GPU.

 a. First, compile and profile the code as-is:

   ```
   module load nsight-systems
   nvcc -o array_inc array_inc.cu
   lsfrun nsys profile --stats=true ./array_inc
   ```
 
   Make a note of the kernel execution duration.
   
 b. Now, modify the code to use managed memory. Replace the malloc operations with cudaMallocManaged, and eliminate the cudaMemcpy operations.  Do you need to replace the *cudaMemcpy* operation from device to host with a *cudaDeviceSynchronize()*? Why? Now, compile and profile the code again. Compare the kernel execution duration to the previous result. Note the profiler indication of CPU and GPU page faults.

 c. Now, modify the code to insert prefetching of the array to the GPU immediately before the kernel call, and back to the CPU immediately after the kernel call. Compile and profile the code again. Compare the kernel execution time to the previous results. Are there still any page faults? Why?
 
 d. Bonus: Modify the code to run the *inc()* kernel 10000 times in a row instead of just once. What can be said about the impact of memory operations on our runtime? What would this suggest for a real-world application?

If you need help, refer to the *array_inc_solution.cu*.
