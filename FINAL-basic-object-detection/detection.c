#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define FILE_NAME "input.txt"
#define WORK_TAG 0
#define TERMINATE_TAG 1

typedef struct od_obj
{
    int id;
    int dim;
    int *data;
} od_obj;

int readObjects(FILE *fp, od_obj **objects, int num_objects)
{
    od_obj *ptr = *objects = (od_obj *)calloc(num_objects, sizeof(od_obj));
    if (!ptr)
    {
        printf("[ERROR] - Memory allocation failed");
        return -1;
    }
    for (int i = 0; i < num_objects; i++)
    {
        fscanf(fp, "%d", &ptr[i].id);
        fscanf(fp, "%d", &ptr[i].dim);
        int obj_size = ptr[i].dim * ptr[i].dim;
        ptr[i].data = (int *)calloc(obj_size, sizeof(int));
        if (!ptr[i].data)
        {
            printf("[ERROR] - Memory allocation failed");
            return -1;
        }
        for (int j = 0; j < obj_size; j++)
            fscanf(fp, "%d", &ptr[i].data[j]);
    }
    return 1;
}

int readFromFile(const char *fileName, double *M, od_obj **images, int *N, od_obj **objs, int *K)
{
    FILE *fp;
    // // Open file for reading data
    if ((fp = fopen(fileName, "r")) == 0)
    {
        printf("[ERROR] - Cannot open file %s for reading\n", fileName);
        return -1;
    }
    fscanf(fp, "%lf", M);
    fscanf(fp, "%d", N);
    if (readObjects(fp, images, *N) < 0)
        return -1;
    fscanf(fp, "%d", K);
    if (readObjects(fp, objs, *K) < 0)
        return -1;
    return 1;
}

int master(int np)
{
    int N, K;
    double M;
    od_obj *objs;
    od_obj *images;
    MPI_Status status;
    if (readFromFile(FILE_NAME, &M, &images, &N, &objs, &K) < 0)
    {
        MPI_Abort(MPI_COMM_WORLD, 0);
        return -1;
    }
    // Send num of objects and match val to slaves
    MPI_Bcast(&M, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Send objects from to slaves
    for (int i = 0; i < K; i++)
    {
        MPI_Bcast(&objs[i].id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&objs[i].dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int obj_size = objs[i].dim * objs[i].dim;
        MPI_Bcast(objs[i].data, obj_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    int current = 0;
    for (int p = 1; p < np && current < N; p++, current++)
    {
        MPI_Send(&images[current].id, 1, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
        MPI_Send(&images[current].dim, 1, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
        int img_size = images[current].dim * images[current].dim;
        MPI_Send(images[current].data, img_size, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
        printf("MASTER SEND TO %d: %d %d - %d %d\n", p, images[current].id, images[current].dim, images[current].data[0], images[current].data[1]);
    }

    return 1;
}

int slave(int rank)
{
    int K;
    double M;
    od_obj *objs;
    MPI_Status status;
    MPI_Bcast(&M, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    objs = (od_obj *)calloc(K, sizeof(od_obj));
    if (!objs)
    {
        printf("[ERROR] - Memory allocation failed");
        MPI_Abort(MPI_COMM_WORLD, 0);
        return -1;
    }
    // Recive objects from master
    for (int i = 0; i < K; i++)
    {
        MPI_Bcast(&objs[i].id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&objs[i].dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int obj_size = objs[i].dim * objs[i].dim;
        objs[i].data = (int *)calloc(obj_size, sizeof(int));
        MPI_Bcast(objs[i].data, obj_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // Recive working tag & image from master
    while (1)
    {
        od_obj img;
        MPI_Recv(&img.id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TERMINATE_TAG)
            break;
        MPI_Recv(&img.dim, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int img_size = img.dim * img.dim;
        img.data = (int *)calloc(img_size, sizeof(int));
        if (!img.data)
        {
            printf("[ERROR] - Memory allocation failed");
            MPI_Abort(MPI_COMM_WORLD, 0);
            return -1;
        }
        MPI_Recv(img.data, img_size, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("SLAVE %d RECIVED: %d %d - %d %d\n", rank, img.id, img.dim, img.data[0], img.data[1]);
        break;
    }

    return 1;
}

int main(int argc, char *argv[])
{
    // init MPI variables
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0)
        master(np);
    else
        slave(rank);

    MPI_Finalize();
    return 0;
}
