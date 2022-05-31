### Summerize an resluts of double array using MPI and OpenMP with 2 proccesses and X threads ###


FILES:
* ex3.c - code for parallel summerize with mpi and omp 
* gen_data.py - code to generate data for the program for given n 
* data.txt - text file containing the floats for process
    

COMPILATION:
> to run the above files mpich must be installed on your machine,
    in project diractory run:
> * ```mpicc search_sub.c -o exec_name -fopenmp```

EXECUTION:
> to execute compiled files you need to run 2 proccess and K max iteration for calculation:
> * ```mpiexec exec_name -np 2 K```
