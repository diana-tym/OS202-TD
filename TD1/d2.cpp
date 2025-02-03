#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size != 4) {
        if (rank == 0) {
            printf("Need 4 processes for d = 2\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        int n = 10;
        MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&n, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        printf("Process %d send n = %d to process 1.\n", rank, n);
        printf("Process %d send n = %d to process 2.\n", rank, n);
    }
    if (rank == 1 || rank == 2) {
        int n;
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received n = %d from process 0.\n", rank, n);

        MPI_Send(&n, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
        printf("Process %d send n = %d to process 3.\n", rank, n);

    }
    if (rank == 3) {
        int n;
        for (int i = 0; i < 2; i++) {
            MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("Process %d received n = %d from processes 1 and 2.\n", rank, n);
    }

    MPI_Finalize();

    return 0;
}
