#pragma once
#include <stdio.h>

#define FILE_NAME "input.txt"
#define WORK_TAG 0
#define TERMINATE_TAG 1
#define SEQUENTIAL 0
#define PARALLEL_THREAD 1
#define PARALLEL_CUDA 2

typedef struct od_obj
{
    int id;
    int dim;
    int *data;
} od_obj;

typedef struct od_res
{
    int obj_id;
    int img_id;
    int dim;
    double *data;
} od_res;


void test(int *data, int n);
int computeOnGPU(int *data, int n);

int readObjects(FILE *fp, od_obj **objects, int num_objects);
int readFromFile(const char *fileName, double *M, od_obj **images, int *N, od_obj **objs, int *K);
int detection(od_obj *objs, int K, od_obj img, double match_value, int exec_type);
int detectionSeqAll(od_obj *objs, int K, od_obj img, double match_value, int activate_omp);
od_res* calculateDiffMatrix(od_obj *img, od_obj *obj, double match_value);
double calc_dif(int x, int y);
int master(int np);
int slave(int rank);
