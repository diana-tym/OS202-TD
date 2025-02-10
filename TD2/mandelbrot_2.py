# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

proc_lignes_n = height // size
local_convergence = np.empty((proc_lignes_n, width), dtype=np.double)

# Calcul de l'ensemble de mandelbrot :

loc_rows = []
loc_res = []

deb = time()
y_list = []
for y in range(rank, height, size):
    y_list.append(y)

    row_res = np.zeros(width)

    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        row_res[x] = mandelbrot_set.convergence(c, smooth=True)

    loc_rows.append(y)
    loc_res.append(row_res)

fin = time()

print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}, rank : {rank}")

convergence = None

if (rank == 0):
    convergence = np.empty((height, width), dtype=np.double)

    for _ in range(size - 1):
        status = MPI.Status()
        num_rows = comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)

        for _ in range(num_rows):
            y, row_data = comm.recv(source=status.source, tag=2)
            convergence[y, :] = row_data

    for y, row_data in zip(loc_rows, loc_res):
        convergence[y, :] = row_data

    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()

else:
    comm.send(len(loc_rows), 0, 1)
    for y, row_data in zip(loc_rows, loc_res):
        comm.send((y, row_data), 0, 2)
