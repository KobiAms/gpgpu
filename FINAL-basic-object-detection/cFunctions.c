#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
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
    double t1 = MPI_Wtime();
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
        // printf("MASTER SEND TO %d: %d %d - %d %d\n", p, images[current].id, images[current].dim, images[current].data[0], images[current].data[img_size-1]);
        fflush(stdout);
        sleep(0.2);

    }

    int res;
    for (int terminated = 1; terminated < np; current++)
    {
        // recive answer from slave
        MPI_Recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // printf("MASTER RECEIVED [%d] RESULT FROM SLAVE %d\n", res, status.MPI_SOURCE);
        fflush(stdout);
        sleep(0.2);

        // if there any other tasks to do
        if (current < N)
        {
            // send task to slave
            MPI_Send(&images[current].id, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            MPI_Send(&images[current].dim, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            int img_size = images[current].dim * images[current].dim;
            MPI_Send(images[current].data, img_size, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            // printf("MASTER SEND TO %d: %d %d - %d %d\n", status.MPI_SOURCE, images[current].id, images[current].dim, images[current].data[0], images[current].data[img_size-1]);
            fflush(stdout);
            sleep(0.2);

        }
        else
        {
            // send terminate tag to slave
            MPI_Send(0, 0, MPI_BYTE, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
            terminated++;
        }
    }
    double t2 = MPI_Wtime();
    printf("Time: %f\n", t2 - t1);

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
        // printf("SLAVE %d RECIVED: %d %d - %d %d\n", rank, img.id, img.dim, img.data[0], img.data[img_size-1]);
        detection(objs, K, img, M, CUDA_ON, OPEN_MP_ON);        
        fflush(stdout);
        sleep(0.2);
        MPI_Send(&img.dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(img.data);
    }

    return 1;
}

double calc_dif(int x, int y)
{
    return abs(((double) x -(double) y) / (double)x);
}

int detection(od_obj *objs, int K, od_obj img, double match_value, int cuda_mode, int omp_mode)
{
    int find = 0;
    od_res_matrix* res_matrix;
    for(int k = 0; k < K; k++){
        
        if(cuda_mode == CUDA_ON){
            res_matrix = calculateDiffCUDA(&img, &objs[k]);
        } else {
            res_matrix = calculateDiffCPU(&img, &objs[k], omp_mode);
        }
        if(res_matrix){
            find = searchValue(res_matrix, match_value, omp_mode);
            free(res_matrix);
        } else {
            printf("Result matrix not received for img: %d, obj: %d\n", objs[k].id, img.dim);
            fflush(stdout);
            sleep(0.2);
        }
        
    }
    
    return find;
}

int searchValue(od_res_matrix *res, double match_value, int omp_mode){

    int find = 0;
    double(*res_2d)[res->dim] = (double(*)[res->dim])res->data;
    omp_set_dynamic(0);
    #pragma omp parallel for
    for (int k = 0; k < res->dim; k++){
        for (int j = 0; j < res->dim; j++){
            
            if(res_2d[k][j] < match_value){
                printf("Picture %d found Objetc %d in Position(%d, %d)\n", res->img_id, res->obj_id, k, j);
                fflush(stdout);
                sleep(0.2);
                find = 1;
            }
        }
    }
    return find;
}

od_res_matrix *calculateDiffCPU(od_obj *img, od_obj *obj, int omp_mode)
{
    od_res_matrix *res = (od_res_matrix *)calloc(1, sizeof(od_res_matrix));
    if (!res)
    {
        printf("[ERROR] - Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
        return NULL;
    }
    res->obj_id = obj->id;
    res->img_id = img->id;
    res->dim = img->dim - obj->dim + 1;

    int res_size = res->dim * res->dim;
    res->data = (double *)calloc(res_size, sizeof(double));
    if (!res->data)
    {
        printf("[ERROR] - Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
        return NULL;
    }

    int(*img_2d)[img->dim] = (int(*)[img->dim])img->data;
    int(*obj_2d)[obj->dim] = (int(*)[obj->dim])obj->data;
    double(*res_2d)[res->dim] = (double(*)[res->dim])res->data;

    omp_set_dynamic(0);
    #pragma omp parallel for
    for (int i = 0; i < res->dim; i++)
        for (int j = 0; j < res->dim; j++)
            for (int k = 0; k < obj->dim; k++)
                for (int l = 0; l < obj->dim; l++)
                    res_2d[i][j] += calc_dif(img_2d[i + k][j + l], obj_2d[k][l]);

    return res;
}
