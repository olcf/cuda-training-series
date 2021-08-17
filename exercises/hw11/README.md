# Multi-Process Service

On Cori GPU, first grab an interactive session. Make sure that you request at least a few slots for MPI, but we'll only need one GPU.

```
module purge
module load cgpu gcc/8.3.0 cuda/11.4.0 openmpi/4.0.3
salloc -A ntrain -q shared --reservation=cuda_mps -C gpu -N 1 -n 4 -t 60 -c 4 --gpus=1
```

The test code used in the lecture is in `test.cu`, and it can be compiled with.

```
nvcc -o test -ccbin=mpicxx test.cu
```

If you're running somewhere where you don't have MPI, you can compile the application without MPI as follows:

```
nvcc -DNO_MPI -o test test.cu
```

Then in all of the examples below, instead of launching with `mpirun`, use the provided `run_no_mpi.sh` script, which launches 4 redundant copies of the same process. This script might also be useful for systems like Summit where you launch jobs from a different node than the compute node, where `nsys jsrun ...` is less useful than `jsrun ... nsys`.

## Verifying the lecture findings

Your exercise is to try some of the experiments from the lecture and see if you can reproduce the findings. Try the following experiments first, without MPS (note that this application does take about 20 seconds to run, so be patient):

```
nsys profile --stats=true -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e9 -f true mpirun -np 1 ./test 1073741824
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_no_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

Verify from both the application stdout and from the profiling data that the average kernel runtime is longer when using 4 ranks on the same GPU.

Now start MPS and repeat the above experiment with 4 ranks, verifying that the average kernel runtime is about the same as in the 1 rank case (again, consult both the stdout and the profiling data).

```
nvidia-cuda-mps-control -d
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_with_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

Now verify that you can stop MPS and the original behavior returns.

```
echo "quit" | nvidia-cuda-mps-control
nsys profile --stats=true -t nvtx,cuda -s none -o 4_ranks_no_MPS_N_1e9 -f true mpirun -np 4 ./test 1073741824
```

## Experimenting with problem size

Vary the problem size `N` until you've found the minimum size where you can definitively say that MPS provides a clear benefit over the default compute mode case.
