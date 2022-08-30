#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__global__ void diffKernel(int** img, int img_dim, int** obj, int obj_dim, double **res, int res_dim)
{
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	int l = blockIdx.x * blockDim.x + threadIdx.x;

	if (k < img_dim && l < img_dim) {
		for (int i = 0; i < obj_dim; i++) {
			for (int j = 0; j < obj_dim; j++) {
                res[i][j] +=  abs(((double) img[i + k][j + l] -(double) obj[k][l]) / (double)img[i + k][j + l]);
			}
		}
	}
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


    int(*img_2d)[img->dim] = (int(*)[img->dim])img_data;
    int(*obj_2d)[obj->dim] = (int(*)[obj->dim])obj_data;
    double(*res_2d)[res->dim] = (double(*)[res->dim])res_data;

    printf("Copy all data to device \n");


    // int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // incrementByOne<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    
    
        // Launch the Kernel
    int threadsPerBlock = 256;
	int blocksPerGrid = ceil(double(res->dim) / double(threadsPerBlock));

	dim3 gridDim(blocksPerGrid, blocksPerGrid);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

    // diffKernel<<<blocksPerGrid, threadsPerBlock>>>()




    

    err1 = cudaFree(res_data);
    err2 = cudaFree(img_data);
    err3 = cudaFree(obj_data);

    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to free allocated memory on device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Free all data to device \n");




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

