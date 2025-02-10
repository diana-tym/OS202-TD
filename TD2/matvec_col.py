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

A_T = A.T
loc_A = np.zeros((loc_n, dim))
loc_u = np.zeros(loc_n)

deb = time()

comm.Scatter(A_T, loc_A, root=0)
comm.Scatter(u, loc_u, root=0)

loc_A = loc_A.T
loc_res = np.dot(loc_A, loc_u)
v = np.zeros(dim)

comm.Allreduce(loc_res, v, MPI.SUM)

fin = time()

if (rank == 0):
    # print(f"Result = {v}")
    pass

print(f"Temps du calcul : {fin-deb},  : {rank}")


