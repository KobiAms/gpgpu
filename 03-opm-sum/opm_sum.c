#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include "mpi.h"
#define NUM_OF_THREADS 2
#define FILE_NAME "input.txt"

// b = max(cos(exp(sin(a * k)))),  for k = 0, 1, 2, …, K
double parallel_calculation_sum(double *A, int N, int K)
{
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_OF_THREADS);
    double sum = 0;
    #pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < N; i++)
    {
        double max = -1;
        for (int k = 0; k <= K; k++)
        {
            double temp_b = cos(exp(sin(k * A[i])));

            if (temp_b > max)
                max = temp_b;
        }
        sum+=max;
    }

    return sum;
}

double *readFromFile(const char *fileName, int *N)
{
    FILE *fp;
    double *data;
    // Open file for reading points
    if ((fp = fopen(fileName, "r")) == 0)
    {
        printf("cannot open file %s for reading\n", fileName);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    // N
    fscanf(fp, "%d", N);
    // Allocate array of points end Read data from the file
    data = (double *)calloc(*N, sizeof(double));
    if (!data)
    {
        printf("Problem to allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    // read line by line
    for (int i = 0; i < (*N); i++)
        fscanf(fp, "%lf", &data[i]);
    // close file descriptor
    fclose(fp);

    return data;
}

int main(int argc, char *argv[])
{

    // init MPI variables
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // get the max iteration value from argv
    int K = atoi(argv[argc - 1]);
    int N;
    double *A, *B;
    if (np != 2)
    {
        printf("Number of process different then 2\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    double t1 = MPI_Wtime();
    // master reads the file
    if (rank == 0)
        A = readFromFile(FILE_NAME, &N);

    // master broadcast N to slave/s
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // any process initial B array and validate
    B = (double *)calloc(N / 2, sizeof(double));
    if (!B)
    {
        printf("Problem to allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    // master scatter the array to process
    if (rank == 0)
        MPI_Scatter(A, N / 2, MPI_DOUBLE, B, N / 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, MPI_DOUBLE, B, N / 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // each process calculate the sum required
    double sum = parallel_calculation_sum(B, N / 2, K);

    // master gather the results and summarize to total result
    if (rank == 0)
    {
        double results[np];
        MPI_Gather(&sum, 1, MPI_DOUBLE, results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        sum = 0;
        for (int i = 0; i < np; i++)
            sum += results[i];
        double t2 = MPI_Wtime();
        printf("Result: %f \tTime: %f\n", sum, t2 - t1);
        free(A);
    }
    else
    {
        // slaves send result to master
        MPI_Gather(&sum, 1, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(B);
    MPI_Finalize();
    return 0;
}
