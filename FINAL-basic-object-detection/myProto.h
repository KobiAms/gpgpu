#pragma once
#include <stdio.h>

#define DATA_DATA_FILE_NAME "input.txt"
#define OUTPUT_FILE_NAME "output.txt"
#define OUTPUT_FILE_NAME "output.txt"
#define WORK_TAG 0
#define TERMINATE_TAG 1
#define CUDA_ON 2
#define CUDA_OFF 3
#define OPEN_MP_ON 4
#define OPEN_MP_OFF 5


typedef struct od_obj
{
    int id;
    int dim;
    int *data;
} od_obj;

typedef struct od_res_matrix_matrix
{
    int obj_id;
    int img_id;
    int dim;
    double *data;
} od_res_matrix;

int master(int np, int OPEN_MP_MODE, int CUDA_MODE);
int slave(int rank, int OPEN_MP_MODE, int CUDA_MODE);


int readObjects(FILE *fp, od_obj **objects, int num_objects);
int readFromFile(const char *fileName, double *M, od_obj **images, int *N, od_obj **objs, int *K);

int detection(od_obj *objs, int K, od_obj img, double match_value, int cuda_mode, int omp_mode, int detection_info[]);

od_res_matrix* calculateDiffCPU(od_obj *img, od_obj *obj, int omp_mode);
int searchValue(od_res_matrix *res, double match_value, int omp_mode, int detection_info[]);
double calc_dif(int x, int y);

od_res_matrix* calculateDiffCUDA(od_obj *img, od_obj *obj);


