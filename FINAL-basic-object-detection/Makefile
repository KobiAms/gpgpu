all: build run clean
runAllModes: runSeq runOmp runCuda runOmpCuda

build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions.o  /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 5 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP

runOmpCuda:
	mpiexec -np 4 ./mpiCudaOpemMP 2 4

runCuda:
	mpiexec -np 4 ./mpiCudaOpemMP 2 5

runOmp:
	mpiexec -np 4 ./mpiCudaOpemMP 3 4

runSeq:
	mpiexec -np 4 ./mpiCudaOpemMP 3 5
