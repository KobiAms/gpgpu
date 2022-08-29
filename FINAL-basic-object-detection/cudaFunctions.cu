#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__  void incrementByOne(int *arr, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Increment the proper value of the arrray according to thread ID 
    if (i < numElements)
        arr[i]++;
}


// int computeOnGPU(int *data, int numElements)
od_res_matrix* calculateDiffCUDA(od_obj *img, od_obj *obj) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    od_res_matrix *res = (od_res_matrix *)calloc(1, sizeof(od_res_matrix));
    if (!res)
    {
        printf("[ERROR] - Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    res->obj_id = obj->id;
    res->img_id = img->id;
    res->dim = img->dim - obj->dim + 1;

    int res_size = res->dim * res->dim;
    res->data = (double *)calloc(res_size, sizeof(double));
    if (!res->data)
    {
        printf("[ERROR] - Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    printf("computeOnGPU start\n");
    // Allocate memory on GPU to copy the data from the host
    int *res_data, *img_data, *obj_data;
    cudaError_t err1 = cudaMalloc((void **)&img_data, img->dim*img->dim);
    cudaError_t err2 = cudaMalloc((void **)&obj_data, obj->dim*obj->dim);
    cudaError_t err3 = cudaMalloc((void **)&res_data, res->dim*res->dim);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    


    // // Copy data from host to the GPU memory
    // err = cudaMemcpy(d_A, data, size, cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }


    // // Launch the Kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // incrementByOne<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // Copy the  result from GPU to the host memory.
    // err = cudaMemcpy(data, d_A, size, cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // printf("computeOnGPU end\n");

    // // Free allocated memory on GPU
    // if (cudaFree(d_A) != cudaSuccess) {
    //     fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    return NULL;
}

