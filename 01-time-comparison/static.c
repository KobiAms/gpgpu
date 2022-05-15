#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define FILE_NAME "points.txt"

// This function simulates heavy computations,
// its run time depends on x, y and param values
// DO NOT change this function!!

double heavy(double x, double y, int param)
{
    double center[2] = {0.4, 0.2};
    int i, loop, size = 1, coeff = 10000;
    double sum = 0, dx, dy, radius = 0.2 * size;
    int longLoop = 1000, shortLoop = 1;
    double pi = 3.14;
    dx = (x - center[0]) * size;
    dy = (y - center[1]) * size;
    loop = (sqrt(dx * dx + dy * dy) < radius) ? longLoop : shortLoop;

    for (i = 1; i < loop * coeff; i++)
        sum += cos(2 * pi * dy * dx + 0.1) * sin(exp(10 * cos(pi * dx))) / i;

    return sum;
}

double *readFromFile(const char *fileName, int *numberOfPoints, int *param)
{
    FILE *fp;
    double *points;

    // Open file for reading points
    if ((fp = fopen(fileName, "r")) == 0)
    {
        printf("cannot open file %s for reading\n", fileName);
        exit(0);
    }

    // Param
    fscanf(fp, "%d", param);

    // Number of points
    fscanf(fp, "%d", numberOfPoints);

    // Allocate array of points end Read data from the file
    points = (double *)malloc(2 * *numberOfPoints * sizeof(double));
    if (points == NULL)
    {
        printf("Problem to allocate memory\n");
        exit(0);
    }
    for (int i = 0; i < *numberOfPoints; i++)
    {
        fscanf(fp, "%le %le", &points[2 * i], &points[2 * i + 1]);
    }

    fclose(fp);

    return points;
}

int main(int argc, char *argv[])
{
    // initial MPI args
    int rank, np;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0)
    {
        ////////////////////////////////////////////////////////////////////////////////////
        //                                                                                //
        //                                  MASTER                                        //
        //                                                                                //
        ////////////////////////////////////////////////////////////////////////////////////

        // get start time
        double t1 = MPI_Wtime();
        // initial relevant params
        double *points, *batch, answer, res;
        int param, numberOfPoints, batch_size, to_cpy;

        // get points
        points = readFromFile(FILE_NAME, &numberOfPoints, &param);
        // calculate each batch size by number of points and number of proccesses
        batch_size = 2 * ceil(numberOfPoints / (np - 1));
        batch = (double *)calloc(batch_size + 1, sizeof(double));
        if (!batch)
        {
            perror("Error in allocation");
            exit(0);
        }
        // insert param number into the batch first value
        batch[0] = (double)param;
        for (int p = 1, curr = 0; p < np; p++)
        {
            // detect how many to send to last proccess
            to_cpy = fmin(batch_size, numberOfPoints * 2 - curr);
            // copy points to memory and send to slave
            memcpy(&batch[1], &points[curr], sizeof(double) * (to_cpy));
            MPI_Send(batch, to_cpy + 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            curr += to_cpy;
        }
        // receive mid result from each slave and calculate final result
        for (int p = 1; p < np; p++)
        {
            MPI_Recv(&res, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            answer = fmax(answer, res);
        }
        free(batch);
        free(points);
        double t2 = MPI_Wtime();
        printf("answer = %e\ntime = %f sec\n", answer, t2 - t1);
    }
    else
    {
        ////////////////////////////////////////////////////////////////////////////////////
        //                                                                                //
        //                                  SLAVE                                         //
        //                                                                                //
        ////////////////////////////////////////////////////////////////////////////////////

        // detect how many tasks receive
        int count;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &count);
        // allocation relevant
        double *points = (double *)calloc(count, sizeof(double));
        MPI_Recv(points, count, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        double answer;
        int param = (double)points[0];
        for (int i = 1; i < count; i += 2)
            answer = fmax(answer, heavy(points[i], points[i + 1], param));
        MPI_Send(&answer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free(points);
    }

    MPI_Finalize();
    return 0;
}
