

FILES:
    static.c - code of static prallalism to do heavy task on points file (fixed size of tasks saparation)
    dynamic.c - code of synamic prallalism to do heavy task on points file (one by one saparation)

COMPILATION:
    to run the above files mpich must be installed on your machine,
    in root diractory run:
                            mpicc file_name.c -o exec_name

EXECUTION:
    to execute compiled files with X (x>=2) workers run:
                                    mpiexec exec_name -np X 
