#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <chrono>

double random_double(unsigned int *seed) {
    return (double)rand_r(seed) / RAND_MAX;
}

int main() {
    long long int num_points = 10000000;
    long long int count_in_circle = 0;

    double start_time = omp_get_wtime();

    #pragma omp parallel reduction(+:count_in_circle)
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num(); 
        
        #pragma omp for
        for (long long int i = 0; i < num_points; i++) {
            double x = random_double(&seed) * 2.0 - 1.0;
            double y = random_double(&seed) * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                count_in_circle++;
            }
        }
    }

    double pi = 4.0 * (double)count_in_circle / (double)num_points;
    double end_time = omp_get_wtime();

    printf("OpenMP: π ≈ %.10f\n", pi);
    printf("Extcution time: %.5f sec\n", end_time - start_time);
    return 0;
}
