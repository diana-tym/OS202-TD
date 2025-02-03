#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int d = log2(size);
    if (pow(2, d) != size) {
        if (rank == 0) {
            std::cerr << "Number of processes must be 2^d." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int M;
    if (rank == 0) {
        M = 10;
        std::cout << "Process " << rank << " set M = " << M << std::endl;
    }

    int mask = (1 << d) - 1;  // (2^d - 1)
    
    double start_time = MPI_Wtime();

    for (int i = d - 1; i >= 0; i--) {
        mask ^= (1 << i);  // (XOR з 2^i)
        if ((rank & mask) == 0) {
            int partner = rank ^ (1 << i);
            if ((rank & (1 << i)) == 0) {
                // Sender
                MPI_Send(&M, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                printf("Process %d send n = %d to process %d.\n", rank, M, partner);
            } else {
                // Receiver
                MPI_Recv(&M, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Process %d received M = %d from process %d.\n", rank, M, partner);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    double total_time;
    MPI_Reduce(&end_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed_time = total_time - start_time;
        std::cout << "Total time: " << elapsed_time << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
