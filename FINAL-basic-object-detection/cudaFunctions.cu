#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__global__ void diffKernel(int* img, int img_dim, int* obj, int obj_dim, double *res, int res_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// float sum = 0.0;

    double x, y, sum = 0;

	if (row < res_dim && col < res_dim) {
		for (int i = 0; i < obj_dim; i++) {
			for (int j = 0; j < obj_dim; j++) {
                x = img[(row + i) * img_dim + col + j];
                y = obj[i * obj_dim + j];
				sum += abs((x-y)/x);
			}
		}
        res[row * res_dim + col] = sum;
	}
}



// int computeOnGPU(int *data, int numElements)
od_res_matrix* calculateDiffCUDA(od_obj *img, od_obj *obj) {

    // Error code to check return values for CUDA calls
    cudaError_t err1 = cudaSuccess, err2 = cudaSuccess, err3 = cudaSuccess;

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
    int img_size = img->dim * img->dim;
    int obj_size = obj->dim * obj->dim;
    res->data = (double *)calloc(res_size, sizeof(double));
    if (!res->data)
    {
        printf("[ERROR] - Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory on GPU to copy the data from the host
    double *d_res_data;
    int *d_img_data, *d_obj_data;
    

    err1 = cudaMalloc((void **)&d_img_data, img_size*sizeof(int));
    err2 = cudaMalloc((void **)&d_obj_data, obj_size*sizeof(int));
    err3 = cudaMalloc((void **)&d_res_data, res_size*sizeof(double));
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s - %s - %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        exit(EXIT_FAILURE);
    }




    // copy data to device
    err1 = cudaMemcpy(d_res_data, res->data, res_size*sizeof(double), cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(d_img_data, img->data, img_size*sizeof(int), cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(d_obj_data, obj->data, obj_size*sizeof(int), cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s - %s - %s\n", cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        exit(EXIT_FAILURE);
    }
    
    // Launch the Kernel
    int block_dim = 32;
	int grid_dim = ceil(double(res->dim) / double(block_dim));

	dim3 threadsPerBlock(block_dim, block_dim);
    dim3 blocksPerGrid(grid_dim, grid_dim);

    cudaStream_t stream;

    err1 = cudaStreamCreate(&stream);
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to initial stream - %s\n", cudaGetErrorString(err1));
        exit(EXIT_FAILURE);
    }

	diffKernel <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (d_img_data, img->dim, d_obj_data, obj->dim, d_res_data, res->dim);
    err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to launch diff kernel -  %s\n", cudaGetErrorString(err1));
        exit(EXIT_FAILURE);
    }

    //  Copy result to host
    err1 = cudaMemcpy(res->data, d_res_data, res_size*sizeof(double), cudaMemcpyDeviceToHost);
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

    return res;
}

