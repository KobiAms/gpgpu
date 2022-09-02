#include <mpi.h>
#include <stdio.h>
// #include <omp.h>
#include <stdlib.h>
#include "myProto.h"

/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/

int main(int argc, char *argv[])
{

    // init MPI variables
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int OMP_MODE = OPEN_MP_ON;
    int CUDA_MODE = CUDA_ON;


    
    if(argc == 3){
        CUDA_MODE = atoi(argv[1]);
        OMP_MODE = atoi(argv[2]);
    }    

    if (rank == 0)
        master(np, OMP_MODE, CUDA_MODE);
    else
        slave(rank, OMP_MODE, CUDA_MODE);

    MPI_Finalize();
    return 0;
}
