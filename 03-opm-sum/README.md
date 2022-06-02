### Summerize an resluts of double array using MPI and OpenMP with 2 proccesses and X threads ###

#### RESULT:

*lscpu: 8 cores - 16 threads*

|    omp threads    |    exec time    |   total threads|              explenation           |
| :---------------: |:---------------:|:--------------:| :---------------------------------:|
|         2         |     0.976332    |        4       |  Too few threads - 12 idle         |
|         4         |     0.557334    |        8       |  Too few threads - 8 idle          |
|         8         |     0.406791    |        16      |  Best fit - 0 idle                 |
|         16        |     0.411738    |        32      |  Too many threads - thread racing  |



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
