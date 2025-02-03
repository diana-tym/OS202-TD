#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size != 2) {
        if (rank == 0) {
            printf("Need 2 processes for d = 1\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        int n = 10;
        MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d send n = %d to process 1.\n", rank, n);
    } else {
        int n;
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received n = %d from process 0.\n", rank, n);
    }

    MPI_Finalize();

    return 0;
}
