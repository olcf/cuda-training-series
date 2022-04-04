# Homework 1

These exercises will have you write some basic CUDA applications. You will learn how to allocate GPU memory, move data between the host and the GPU, and launch kernels.

## **1. Hello World**

Your first task is to create a simple hello world application in CUDA. The code skeleton is already given to you in `hello.cu`. Edit that file, paying attention to the FIXME locations, so that the output when run is like this:

```
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```

(the ordering of the above lines may vary; ordering differences do not indicate an incorrect result)

Note the use of `cudaDeviceSynchronize()` after the kernel launch. In CUDA, kernel launches are *asynchronous* to the host thread. The host thread will launch a kernel but not wait for it to finish, before proceeding with the next line of host code. Therefore, to prevent application termination before the kernel gets to print out its message, we must use this synchronization function.

After editing the code, compile it using the following:

```
module load cuda
nvcc -o hello hello.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. `nvcc` is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

If you have trouble, you can look at `hello_solution.cu` for a complete example.

To run your code at OLCF on Summit, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./hello
```

Alternatively, you may want to create an alias for your `bsub` command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./hello
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --gres=gpu:1 -c 10 ./hello
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance until January 18, 2020. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes, you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active:

```
salloc -C gpu -N 1 -t 60 -A m3502 --gres=gpu:1 -c 10
srun -n 1 ./hello
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.

## **2. Vector Add**

If you're up for a challenge, see if you can write a complete vector add program from scratch. Or if you prefer, there is a skeleton code given to you in `vector_add.cu`. Edit the code to build a complete vector_add program. Compile it and run it similar to the method given in exercise 1. You can refer to `vector_add_solution.cu` for a complete example.

Note that this skeleton code includes something we didn't cover in lesson 1: CUDA error checking. Every CUDA runtime API call returns an error code. It's good practice (especially if you're having trouble) to rigorously check these error codes. A macro is given that will make this job easier. Note the special error checking method after a kernel call.

Typical output when complete would look like this:
```
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
```

## **3. Matrix Multiply (naive)**

A skeleton naive matrix multiply is given to you in `matrix_mul.cu`. See if you can complete it to get a correct result. If you need help, you can refer to `matrix_mul_solution.cu`.

This example introduces 2D threadblock/grid indexing, something we did not cover in lesson 1. If you study the code you will probably be able to see how it is a structural extension from the 1D case.

This code includes built-in error checking, so a correct result is indicated by the program.
