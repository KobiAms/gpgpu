#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define FILE_NAME "points.txt"
#define TASK_SIZE 3
#define WORK_TAG 0
#define TERMINATE_TAG 1

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
    int rank, np;
    double task[TASK_SIZE], res;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Master
    if (rank == 0)
    {
        double t1 = MPI_Wtime(), *points, answer = 0;
        int param, numberOfPoints = 10, current = 0;

        points = readFromFile(FILE_NAME, &numberOfPoints, &param);
        task[0] = (double)param;
        // Send task to each of the proccesses
        for (int p = 1; p < np; p++, current += 2)
        {
            memcpy(&task[1], &points[current], sizeof(double) * (TASK_SIZE - 1));
            MPI_Send(task, TASK_SIZE, MPI_DOUBLE, p, WORK_TAG, MPI_COMM_WORLD);
        }
        for (int terminated = 1; terminated < np; current += 2)
        {
            // recive answer from slave
            MPI_Recv(&res, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // if there any other tasks to do
            if (current < numberOfPoints * 2)
            {
                // send task to slave
                memcpy(&task[1], &points[current], sizeof(double) * (TASK_SIZE - 1));
                MPI_Send(task, TASK_SIZE, MPI_DOUBLE, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            }
            else
            {
                // send terminate tag to slave
                MPI_Send(0, 0, MPI_BYTE, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
                terminated++;
            }
            answer = fmax(answer, res);
        }
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
        while (1)
        {
            // recive a msg
            MPI_Recv(task, TASK_SIZE, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // if tag is terminated -> go to sleep
            if (status.MPI_TAG == TERMINATE_TAG)
                break;
            // do work and send to master
            res = heavy(task[1], task[2], (int)task[0]);
            MPI_Send(&res, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}
