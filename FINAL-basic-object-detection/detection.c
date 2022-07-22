#include <stdio.h>
#include <stdlib.h>
#define FILE_NAME "input.txt"

typedef struct od_obj
{
    int id;
    int dim;
    int *data;
} od_obj;

int readObjects(FILE *fp, od_obj** objects, int num_objects){
    od_obj *ptr = *objects = (od_obj *)calloc(num_objects, sizeof(od_obj));
    if(!ptr){
        printf("[ERROR] - Memory allocation failed");
        return -1;
    }
    for (int i = 0; i < num_objects; i++){
        fscanf(fp, "%d", &ptr[i].id);
        fscanf(fp, "%d", &ptr[i].dim);
        int obj_size = ptr[i].dim*ptr[i].dim;
        ptr[i].data = (int *)calloc(obj_size, sizeof(int));
        if(!ptr[i].data){
            printf("[ERROR] - Memory allocation failed");
            return -1;
        }
        for(int j = 0; j < obj_size; j++)
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
    if(readObjects(fp, images, *N) < 0) return -1;
    fscanf(fp, "%d", K);
    if(readObjects(fp, objs, *K) < 0) return -1;
    return 1;
}

int main(int argc, char *argv[])
{
    int N, K;
    double M;
    od_obj *images, *objs;
    
    if(readFromFile(FILE_NAME, &M, &images, &N, &objs, &K) < 0)
        return -1;
    

    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d %d -- %d %d\n", images[i].id, images[i].dim, images[i].data[0], images[i].data[1]);
    // }

    // for (int i = 0; i < K; i++)
    // {
    //     printf("%d %d -- %d %d\n", objs[i].id, objs[i].dim, objs[i].data[0], objs[i].data[1]);
    // }
    
    return 0;
}

