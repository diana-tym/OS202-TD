from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1200

t_total_start = MPI.Wtime()

# Process 0 generates the entire array
if rank == 0:
    all_values = np.random.randint(0, 5000, size=N, dtype=np.int32)
else:
    all_values = None

local_sz = N // size
local_data = np.empty(local_sz, dtype=np.int32)
comm.Scatter(all_values, local_data, root=0)   # send batches of array to processes

local_data.sort()

# Choose the itervals for buckets

midpoints = (local_data[:-1] + local_data[1:]) // 2  
local_intervals = np.concatenate(([local_data[0]], midpoints, [local_data[-1]]))

local_intervals_size = np.array([len(local_intervals)], dtype=np.int32)

if rank == 0:
    all_sizes = np.empty(size, dtype=np.int32)
else:
    all_sizes = None

comm.Gather(local_intervals_size, all_sizes, root=0)

if rank == 0:
    displs = np.insert(np.cumsum(all_sizes), 0, 0)[:-1]
    gathered_intervals = np.empty(sum(all_sizes), dtype=np.int32)
else:
    gathered_intervals = None
    displs = None

comm.Gather(local_intervals, gathered_intervals, root=0)

bucket_intervals = None
if rank == 0:
    gathered_intervals.sort()
    bucket_intervals = gathered_intervals[np.linspace(0, len(gathered_intervals)-1, size+1, dtype=np.int32)]

# Send bucket intervals to all processes
bucket_intervals = comm.bcast(bucket_intervals, root=0)

# Redistribution of numbers into buckets
buckets = [[] for _ in range(size)]
for num in local_data:
    idx = np.searchsorted(bucket_intervals, num, side="right") - 1
    idx = min(idx, size - 1)
    buckets[idx].append(num)
send_counts = np.array([len(b) for b in buckets], dtype=np.int32)
recv_counts = np.empty(size, dtype=np.int32)

comm.Alltoall(send_counts, recv_counts)

send_buf = np.concatenate(buckets)
recv_buf = np.empty(sum(recv_counts), dtype=np.int32)
send_displs = np.insert(np.cumsum(send_counts), 0, 0)[:-1]
recv_displs = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]

comm.Alltoallv([send_buf, send_counts, send_displs, MPI.INT], 
                [recv_buf, recv_counts, recv_displs, MPI.INT])

# Sort the numbers in the bucket
recv_buf.sort()

final_counts = np.array([len(recv_buf)], dtype=np.int32)

if rank == 0:
    gathered_counts = np.empty(size, dtype=np.int32)
else:
    gathered_counts = None

comm.Gather(final_counts, gathered_counts, root=0)

if rank == 0:
    final_displs = np.insert(np.cumsum(gathered_counts), 0, 0)[:-1]
    sorted_data = np.empty(sum(gathered_counts), dtype=np.int32)
else:
    final_displs = None
    sorted_data = None

comm.Gatherv(recv_buf, [sorted_data, gathered_counts, final_displs, MPI.INT], root=0)

t_total_end = MPI.Wtime()
t_total = t_total_end - t_total_start

if rank == 0:
    print(f"N = {N}")
    print(f"Valeurs initiale : {all_values}")
    print(f"Valeurs triees : {sorted_data}")
    print(f"Temps de calcul: {t_total:.6f} seconds")
