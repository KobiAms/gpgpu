### Search Substring in strings array using cartesian matrix and K^2 proccesses ###

FILES:
* serach_sub.c - code for serahc substring in cartesian matrix
* gen_data.py - code to generate data for the program for given k, n, max vars
* data.txt - text file containing the string for each proccess, also includes K, N & max iteraation vars
    

COMPILATION:
> to run the above files mpich must be installed on your machine,
    in project diractory run:
> * ```mpicc search_sub.c -o exec_name```

EXECUTION:
> to execute compiled files you need to run K^2 proccess (K defined in data.txt):
> * ```mpiexec exec_name -np (K*K)```
