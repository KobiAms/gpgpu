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

int master(int np, int OPEN_MP_MODE, int CUDA_MODE)
{
    double t1 = MPI_Wtime();
    int N, K;
    double M;
    od_obj *objs;
    od_obj *images;
    MPI_Status status;
    if (readFromFile(DATA_FILE_NAME, &M, &images, &N, &objs, &K) < 0)
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
    int current = 0, terminated = 1;
    for (int p = 1; p < np; p++)
    {
        if(current < N){
            MPI_Send(&images[current].id, 1, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
            MPI_Send(&images[current].dim, 1, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
            int img_size = images[current].dim * images[current].dim;
            MPI_Send(images[current].data, img_size, MPI_INT, p, WORK_TAG, MPI_COMM_WORLD);
            fflush(stdout);
            sleep(0.2);
            current++;
        } else {
            // send terminate tag to slave
            MPI_Send(0, 0, MPI_BYTE, p, TERMINATE_TAG, MPI_COMM_WORLD);
            terminated++;
        }
    }
    FILE *out_file = fopen(OUTPUT_FILE_NAME, "w+");
    if(!out_file){
        printf("FAILED to open file for output\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    int res;
    for (; terminated < np; current++)
    {
        int detection_info[5] = {0};
        // recive answer from slave
        MPI_Recv(detection_info, 5, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(detection_info[0]){
            fprintf(out_file, "Picture %d found Objetc %d in Position(%d, %d)\n", detection_info[1], detection_info[2], detection_info[3], detection_info[4]);
        } else {
            fprintf(out_file, "Picture %d No Objects were found\n", detection_info[1]);
        }

        // if there any other tasks to do
        if (current < N)
        {
            // send task to slave
            MPI_Send(&images[current].id, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            MPI_Send(&images[current].dim, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
            int img_size = images[current].dim * images[current].dim;
            MPI_Send(images[current].data, img_size, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);

        }
        else
        {
            // send terminate tag to slave
            MPI_Send(0, 0, MPI_BYTE, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
            terminated++;
        }
    }
    fclose(out_file);
    for (int i = 0; i < K; i++)
        free(objs[i].data);
    for (int i = 0; i < N; i++)
        free(images[i].data);
    free(objs);
    free(images);

    double t2 = MPI_Wtime();
    char omp_mode_str[5] = "ON";
    char cuda_mode_str[5] = "ON";
    if(OPEN_MP_MODE == OPEN_MP_OFF){
        omp_mode_str[1] = 'F';
        omp_mode_str[2] = 'F';
        omp_mode_str[3] = '\0';
    } 
    if(CUDA_MODE == CUDA_OFF){
        cuda_mode_str[1] = 'F';
        cuda_mode_str[2] = 'F';
        cuda_mode_str[3] = '\0';
    } 
    printf("OpenMP Node: [%s], \tCuda Mode: [%s], \tRuntime: %f\n", omp_mode_str, cuda_mode_str, t2 - t1);

    return 1;
}

int slave(int rank, int OPEN_MP_MODE, int CUDA_MODE)
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
        int detection_info[5] = {0};
        detection_info[1] = img.id;
        detection(objs, K, img, M, CUDA_MODE, OPEN_MP_MODE, detection_info);       
        MPI_Send(detection_info, 5, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(img.data);
    }
    for (int i = 0; i < K; i++)
        free(objs[i].data);
    free(objs);

    return 1;
}

double calc_dif(int x, int y)
{
    return abs(((double) x -(double) y) / (double)x);
}

int detection(od_obj *objs, int K, od_obj img, double match_value, int cuda_mode, int omp_mode, int detection_info[])
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
            find = searchValue(res_matrix, match_value, omp_mode, detection_info);
            free(res_matrix->data);
            free(res_matrix);
            if(find){
                break;
            }
            
        }
    }
    return find;
}

int searchValue(od_res_matrix *res, double match_value, int omp_mode, int detection_info[]){

    int find = 0;
    double(*res_2d)[res->dim] = (double(*)[res->dim])res->data;
    omp_set_dynamic(0);
    #pragma omp parallel for
    for (int k = 0; k < res->dim; k++){
        for (int j = 0; j < res->dim; j++){
            if(res_2d[k][j] < match_value){
                detection_info[0] = 1;
                detection_info[2] = res->obj_id;
                detection_info[3] = k;
                detection_info[4] = j;
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
    if(omp_mode == OPEN_MP_OFF){
        omp_set_num_threads(1);
    } else {
        omp_set_dynamic(0);
    }
    #pragma omp parallel for
    for (int i = 0; i < res->dim; i++)
        for (int j = 0; j < res->dim; j++)
            for (int k = 0; k < obj->dim; k++)
                for (int l = 0; l < obj->dim; l++)
                    res_2d[i][j] += calc_dif(img_2d[i + k][j + l], obj_2d[k][l]);

    return res;
}
