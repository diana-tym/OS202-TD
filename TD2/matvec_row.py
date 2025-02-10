# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème (peut-être changé)
dim = 1200

if (rank == 0):
    # Initialisation de la matrice
    A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
    # print(f"A = {A}")

    # Initialisation du vecteur u
    u = np.array([i+1. for i in range(dim)])
    # print(f"u = {u}")

else:
    A = np.empty((dim, dim))
    u = np.zeros(dim)

loc_n = dim // size

loc_A = np.zeros((loc_n, dim))

deb = time()

comm.Scatter(A, loc_A, root=0)
comm.Bcast(u, root=0)

loc_res = loc_A.dot(u)
v = np.zeros(dim)

comm.Allgather(loc_res, v)

fin = time()

if (rank == 0):
    # print(f"Result = {v}")
    pass

print(f"Temps du calcul : {fin-deb},  : {rank}")
