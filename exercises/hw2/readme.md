# Homework 2

These exercises will help reinforce the concept of Shared Memory on the GPU.

## **1. 1D Stencil Using Shared Memory**

Your first task is to create a 1D stencil application that uses shared memory. The code skeleton is provided in *stencil_1d.cu*. Edit that file, paying attention to the FIXME locations. The code will verify output and report any errors.

After editing the code, compile it using the following:

```
module load cuda
nvcc -o stencil_1d stencil_1d.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./stencil_1d
```

Alternatively, you may want to create an alias for your *bsub* command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./stencil_1d
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./stencil_1d
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./stencil_1d
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

If you have trouble, you can look at *stencil_1d_solution* for a complete example.

## **2. 2D Matrix Multiply Using Shared Memory**

Next, let's apply shared memory to the 2D matrix multiply we wrote in Homework 1. FIXME locations are provided in the code skeleton in *matrix_mul_shared.cu*. See if you can successfully load the required data into shared memory and then appropriately update the dot product calculation. Compile and run your code using the following:

```
module load cuda
nvcc -o matrix_mul matrix_mul_shared.cu
lsfrun ./matrix_mul
```

Note that timing information is included. Go back and run your solution from Homework 1 and observe the runtime. What runtime impact do you notice after applying shared memory to this 2D matrix multiply? How does it differ from the runtime you observed in your previous implementation?

If you have trouble, you can look at *matrix_mul_shared_solution* for a complete example.
