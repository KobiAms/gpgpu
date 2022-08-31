#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__global__ void diffKernel(int* img, int img_dim, int* obj, int obj_dim, double *res, int res_dim)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	

	if (x < res_dim && y < res_dim) {
		for (int i = 0; i < obj_dim; i++) {
			for (int j = 0; j < obj_dim; j++) {
                res[x*res_dim + y] +=  1;
                // abs(((double) img[(x+i)*img_dim + y + j] -(double) obj[i*obj_dim+j]) / (double) img[(x+i)*img_dim + y + j]);
			}
		}
	}
}



// int computeOnGPU(int *data, int numElements)
od_res_matrix* calculateDiffCUDA(od_obj *img, od_obj *obj) {
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

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
    double *d_res_data;
    int *d_img_data, *d_obj_data;
    cudaError_t err1 = cudaMalloc((void **)&d_img_data, img->dim*img->dim);
    cudaError_t err2 = cudaMalloc((void **)&d_obj_data, obj->dim*obj->dim);
    cudaError_t err3 = cudaMalloc((void **)&d_res_data, res->dim*res->dim);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s - %s - %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        exit(EXIT_FAILURE);
    }

    printf("Copy all data to device \n");

    // copy data to device
    err1 = cudaMemcpy(d_res_data, res->data, res->dim*res->dim, cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(d_img_data, img->data, img->dim*img->dim, cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(d_obj_data, obj->data, obj->dim*obj->dim, cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s - %s - %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        exit(EXIT_FAILURE);
    }
    
    // Launch the Kernel
    int threadsPerBlock = 256;
	int blocksPerGrid = ceil(double(res->dim) / double(threadsPerBlock));

	dim3 gridDim(blocksPerGrid, blocksPerGrid);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

    diffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img_data, img->dim, d_obj_data, obj->dim, d_res_data, res->dim);
    err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to launch diff kernel -  %s\n", cudaGetErrorString(err1));
        exit(EXIT_FAILURE);
    }

    //  Copy result to host
    err1 = cudaMemcpy(res->data, d_res_data, res->dim*res->dim, cudaMemcpyDeviceToHost);
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to copy result matrix from device to host   -  %s\n", cudaGetErrorString(err1));
        exit(EXIT_FAILURE);
    }

    // // Free allocated memory on GPU
    err1 = cudaFree(d_res_data);
    err2 = cudaFree(d_img_data);
    err3 = cudaFree(d_obj_data);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to free allocated memory on device - %s - %s - %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        exit(EXIT_FAILURE);
    }
    printf("Free all data to device \n");

    // Copy the  result from GPU to the host memory.



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

    
    // if (cudaFree(d_A) != cudaSuccess) {
    //     fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    return res;
}

