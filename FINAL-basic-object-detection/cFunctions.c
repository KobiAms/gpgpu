#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "myProto.h"

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
    printf("MASTER: read input - match value: %f, %d images, %d object.\n", M, N, K);
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

    int res;
    for (int terminated = 1; terminated < np; current++)
    {
        // recive answer from slave
        MPI_Recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("MASTER RECEIVED %d FROM SLAVE %d\n", res, status.MPI_SOURCE);
        // if there any other tasks to do
        if (current < N)
        {
            // send task to slave
            MPI_Send(&images[current].id, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            MPI_Send(&images[current].dim, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            int img_size = images[current].dim * images[current].dim;
            MPI_Send(images[current].data, img_size, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            printf("MASTER SEND TO %d: %d %d - %d %d\n", status.MPI_SOURCE, images[current].id, images[current].dim, images[current].data[0], images[current].data[1]);
        }
        else
        {
            // send terminate tag to slave
            MPI_Send(0, 0, MPI_BYTE, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
            terminated++;
        }
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
        printf("[ERROR] - Memory allocation failed\n");
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
            printf("[ERROR] - Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
            return -1;
        }
        MPI_Recv(img.data, img_size, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // detection(objs, K, img, M, SEQUENTIAL);
        printf("SLAVE %d RECIVED: %d %d - %d %d\n", rank, img.id, img.dim, img.data[0], img.data[1]);
        MPI_Send(&img.dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(img.data);
    }

    return 1;
}

double calc_dif(int x, int y)
{
    return abs((x - y) / x);
}

int detection(od_obj *objs, int K, od_obj img, int match_value, int exec_type)
{
    switch (exec_type)
    {
    case SEQUENTIAL:
        detectionSeqAll(objs, K, img, match_value);
        break;
    case PARALLEL_THREAD:

        break;
    case PARALLEL_CUDA:

        break;
    }
    return 1;
}

int detectionSeqAll(od_obj *objs, int K, od_obj img, int match_value)
{
    for (int i = 0; i < K; i++)
    {
        detectionSeq(&img, &objs[i], match_value);
    }
    return 1;
}

od_res *detectionSeq(od_obj *img, od_obj *obj, int match_value)
{
    od_res *res = (od_res *)calloc(1, sizeof(od_res));
    if (!res)
    {
        printf("[ERROR] - Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
        return NULL;
    }

    res->dim = img->dim - obj->dim + 1;
    res->data = (double *)calloc(res->dim * res->dim, sizeof(double));

    // float(*somethingAsMatrix)[2] = (float(*)[2])matrixReturnAsArray;

    int(*img_2d)[img->dim] = (int(*)[img->dim])img->data;
    int(*obj_2d)[obj->dim] = (int(*)[obj->dim])obj->data;
    double(*res_2d)[res->dim] = (double(*)[res->dim])res->data;

    for (int i = 0; i < res->dim; i++)
    {
        for (int j = 0; j < res->dim; j++)
        {

            for (int k = 0; k < obj->dim; k++)
            {
                for (int l = 0; l < obj->dim; l++)
                {
                    res_2d[i][j] += calc_dif(img_2d[i + k][j + l], obj_2d[k][j]);
                    printf("[%d - %d] ", img_2d[i + k][j + l], obj_2d[k][j]);
                }
                printf("\n%lf \n", res_2d[i][j]);
                break;
            }
            break;
        }
        break;
    }

    return NULL;
}
