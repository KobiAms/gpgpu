#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#define NUM_OF_THREADS 2
#define FILE_NAME "data.txt"

// b = max(cos(exp(sin(a * k)))),  for k = 0, 1, 2, …, K

double calculation(double a, int K)
{
    double max = -1;
    for (int k = 0; k < K; k++)
    {
        double temp_b = cos(exp(sin(k * a)));
        if (temp_b > max)
            max = temp_b;
    }

    return max;
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
    for (int i = 0; i < (*N); i++)
    {
        fscanf(fp, "%lf", &data[i]);
    }
    fclose(fp);

    return data;
}

int main(int argc, char *argv[])
{
    int rank, np;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int K = atoi(argv[argc - 1]);
    int N;
    double *A, *B;
    if (np != 2)
    {
        printf("Num of process must be two!!\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    if (rank == 0)
    {
        A = readFromFile(FILE_NAME, &N);
        B = (double *)calloc(N / 2, sizeof(double));
        if (!B)
        {
            printf("Problem to allocate memory\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(A, N / 2, MPI_DOUBLE, B, N / 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N / 2; i++)
        {
            printf("MASTER: %f\n", B[i]);
        }

        free(A);
    }
    else
    {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        B = (double *)calloc(N / 2, sizeof(double));
        if (!B)
        {
            printf("Problem to allocate memory\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        MPI_Scatter(NULL, 0, MPI_DOUBLE, B, N / 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < N / 2; i++)
        {
            printf("SLAVE: %f\n", B[i]);
        }
    }
    free(B);
    MPI_Finalize();
    return 0;
}
