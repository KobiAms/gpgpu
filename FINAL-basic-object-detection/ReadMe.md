
### Summerize an resluts on detecting an object whitin an image using openMPI openMP and Cuda ###


#### **Workflow:**

**Program argv:**
* [**-np**] - number of MPI processes to start (required)
* [**Open MP on/off**] - turn [on(4) / off(5)] openMP (multithreading) behavior(Default - on)
* [**CUDA on/off**] - turn [on(2) / off(3)] CUDA (GPU) behavior (Default - on).

#### **MAIN RESULT**:

*lscpu: 8 cores - 16 threads*


|    Open MP Mode  |    CUDA Mode    |      exec time     |
| :---------------:|:---------------:|:------------------:|
|        [OFF]     |      [OFF]      |     31.926478 sec  |
|        [ON]      |      [OFF]      |      6.689517 sec  |
|        [OFF]     |      [ON]       |      1.012720 sec  |
|        [ON]      |      [ON]       |      0.969275 sec  |


##### **Master:** #####

1. starting program with [*np*] processes (MPI -np argument).
2. start 1 process (rank 0) to be master and [*np-1*] slaves.
3. Master read the input file (input.txt) and broadcasting the match value & the objects (filters) to the slaves.
4. using dynamic behavier, the master sends to the slaves picture by picture to search detection of one of the objects inside the picture and get the result back.
5. Result object int array[5] of: **[*Found Flag, Image ID, Object ID, X detect coordinate, Y detect coordinate*]**.
6. Write the result of each of the images into the output file (otuput.txt).

##### **Slave:** #####

1. Receiving using Broadcast the match value and the objects (filters) from the master
2. Receiving an id of picture and a working/terminate flag from master.
- -  in case the flag is terminate -> free memory and finish program.
- -  in case the flag is work -> continue (go to stage 3).
3. Receive image dimention and image data from master.
4. for each object 
4. Calculate Result matrix. - modes: CPU Sequential, CPU Multithreaded (openMP), GPU (Cuda).
5. Search result within the result matrix. - modes: CPU Sequential, CPU Multithreaded (openMP).
6. send the master the result of the objects detection
7. Go to stage 2


##### FILES:
* cFunctons.c - master, slave and CPU detection algorithm functions.  
* cudaFunctons.cu - detections algorithm with GPU acceleration, and kernel definition.
* main.c - receiving MPI execution arguments and split program into master/slave rules.  
* mtProto.c - functions definitions
* input.txt - text file containing the imgaes and the objects
* Makefile - compilation, execution and cleaning.
    

##### COMPILATION:
> to run the above files mpich must be installed on your machine,
    in project diractory run:
> * ```make build```

##### EXECUTION:
> to execute compiled files you can run any of the commands below :
> * ```make run``` - Cuda & Omp On
> * ```make runOmpCuda``` - same as *make run*
> * ```make runOmp``` - Cuda off, Omp on
> * ```make runCuda``` - Cuda on, Omp off
> * ```make runSeq``` - Cuda off, omp off






