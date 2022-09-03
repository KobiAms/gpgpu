
### Summerize an resluts on detecting an object whitin an image using openMPI openMP and Cuda ###

#### RESULT:

*lscpu: 8 cores - 16 threads*


|    Open MP Mode  |    CUDA Mode    |    exec time   |
| :---------------:|:---------------:|:--------------:|
|        [OFF]     |      [OFF]      |     31.926478  |
|        [ON]      |      [OFF]      |      6.689517  |
|        [OFF]     |      [ON]       |      1.012720  |
|        [ON]      |      [ON]       |      0.969275  |




##### FILES:
* ex3.c - code for parallel summerize with mpi and omp 
* gen_data.py - code to generate data for the program for given n 
* input.txt - text file containing the floats for process
    

##### COMPILATION:
> to run the above files mpich must be installed on your machine,
    in project diractory run:
> * ```mpicc search_sub.c -o exec_name -fopenmp```

##### EXECUTION:
> to execute compiled files you need to run 2 proccess and K max iteration for calculation:
> * ```mpiexec exec_name -np 2 K```






