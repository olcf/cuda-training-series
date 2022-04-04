This excercise, in 3 parts, is designed to walk you through a Nsight Compute-driven analysis-driven optimization sequence. The overall exercise is focused on optimizing square matrix transpose.  This operation can be simply described as:

Bij = Aji

for input matrix A, output matrix B, and indices i and j varying over the square matrix side dimension.  This algorithm involves no compute activity, therefore it is a memory bound algorithm, and our final objective will be to come as close as possible to the available memory bandwidth of the GPU we are running on.

## **1. Naive Global-Memory Matrix Transpose**

For your first task, change into the *task1* directory. There you should edit the *task1.cu* file to complete the matrix transpose operation. Most of the code is written for you, but replace the **FIXME** entries with the proper code to complete the matrix transpose using global memory. The formula given above should guide your efforts. Here are some hints:

 - Each thread reads from (row, col) and writes to (col, row)
 - Using indexing macro:

```cpp
#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) ) 
ld = leading dimension (width)
```

If you need help, you can refer to the *task1_solution.cu* file.  Then compile and test your code:

```bash
module load cuda
./build_nvcc
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

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

You should get a PASS result indication, along with a measurement of performance in terms of achieved bandwidth.

One you have a PASS result, begin the first round of analysis by running the profiler:

```bash
module load nsight-compute
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./task1
```

Here's a breakdown of the metrics we are requesting from the profiler:

 - *l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum*: The number of global load transactions
 - *l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum*: The number of global load requests
 - *l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio*: The number of global load transactions per request
 - *smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct*: The global load efficiency
 - *l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum*: The number of global store transactions
 - *l1tex__t_requests_pipe_lsu_mem_global_op_st.sum*: The number of global store requests
 - *l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio*: The number of global store transactions per request
 - *smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct*: The global store efficiency

Using these metrics, we can easily observe various characteristics of our kernel. Many of these metrics are self-explanatory, but it may not be immediately obvious how global load and store *efficiency* is calculated. We can also calculate our global load and store efficiences by dividing the theoretical minimum number of transactions per request by the actual number of transactions per request we calculated from the above metrics. 

How do we know what the theoretical minimum number of transactions per request actually is? A cache line is 128 bytes, and there are 32 threads in a warp. If the 32 threads are accessing consecutive 4 byte words (i.e. single precision floats), then there should be 4 transactions in that request (we are just asking for four consecutive 32-byte sectors of DRAM). In our case, we are using double precision floats, so the 32 threads would be accessing consecutive 8 byte words (256 bytes total). Therefore, the theoretical minimum number of transactions per request in our case would be 8 (eight consecutive 32-byte sectors of DRAM).

Considering the output of the profiler, are the Global Load Efficiency and Global Store Efficiency both at 100%? Why or why not? This may be a good time to study the load and store indexing carefully, and review the global coalescing rules learned in Homework 4.

## **2. Fix Global Memory Coalescing Issue via Shared-Memory Tiled Transpose**

In task 1, we learned that the naive global memory transpose algorithm suffers from the fact that either the load or store operation must be un-coalesced, i.e. columnar memory access. To fix this, we must come up with a procedure that will allow both coalesced loads and coalesced stores to global memory. Therefore, we will move tiles from the input matrix into shared memory, and then write that tile out to the output matrix, to allow a transpose of the tile. This involves a read from global, write to shared, followe by a read from shared, write to global.

During these two steps, we will need to:

- perform an "*in-tile*" transpose, i.e. either read row-wise and write column-wise, or vice versa
- perform a "*tile-position*" transpose, meaning an input tile at tile indices *i,j* must be stored to tile indices *j,i* in the output matrix.

Change from directory *task1* to *task2*.  Edit the *task2.cu* file, wherever the **FIXME** occurs, to achieve the above two operations. If you need help, refer to the *task2_solution.cu* file. This is the hardest programming assignment of the 3 tasks in this exercise.

As in task1, compile and run your code:

```bash
./build_nvcc
lsfrun ./task2
```

You should get a PASS output.  Has the measured bandwidth improved?

Once again we will use the profiler to help explain our observations.  We have introduced shared memory operations into our algorithm, so we will include shared memory measure metrics in our profiling:

```bash
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct ./task2
```

 - *l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum*: The number of shared load transactions
 - *l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum*: The number of shared store transactions
 - *l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum*: The number of shared load bank conflicts
 - *l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum*: The number of shared store bank conflicts
 - *smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct*: Shared Memory efficiency

You should be able to confirm that the previous global load/global store efficiency issues have been resolved, with proper coalescing.  However now we have a problem with shared memory: bank conflicts.  Review module 4 information on bank conflicts, for a basic definition of how these arise during shared memory access.

## **3. Fixing shared memory bank conflicts**

Our strategy to fix shared memory bank conflicts in this case is fairly simple. We will leave the shared memory indexing unchanged from exercise 2, but we will add a column to the shared memory definition in our code. This will allow both row-wise and columnar access to shared memory (needed for our in-tile transpose step) without bank conflicts.

Change to the *task3* directory.

Modify the *task3.cu* code as needed. If you need help, refer to *task3_solution.cu*.

Compile and run your code in a similar fashion to the previous 2 tasks.

You should get a passing result. Has the achieved bandwidth improved?

You can profile your code to confirm that we are now using shared memory in an efficient fashion, for both loads and stores.

```bash
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct  ./task3
```

Finally, if you wish, compare the achieved bandwidth reported by your code, to a proxy measurement of the peak achievable bandwidth, by running the bandwidthTest CUDA sample code and using the device-to-device memory number for comparison.
