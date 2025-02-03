from mpi4py import MPI
import random

def approximate_pi(nb_samples):
    nb_darts = 0
    for _ in range(nb_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x * x + y * y <= 1:
            nb_darts += 1
    return 4.0 * nb_darts / nb_samples

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

total_samples = 10000000
local_samples = total_samples // size

start_time = MPI.Wtime()
local_pi = approximate_pi(local_samples)
end_time = MPI.Wtime()

local_elapsed = end_time - start_time

max_elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)

global_pi = comm.reduce(local_pi, op=MPI.SUM, root=0)

if rank == 0:
    global_pi /= size
    print(f"mpi4py: π ≈ {global_pi}")
    print(f"Execution time: {max_elapsed} sec")
    