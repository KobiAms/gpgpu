#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define FILE_NAME "data.txt"

char *readFromFile(const char *fileName, int *knm, char **sub)
{
    FILE *fp;
    char *strings;

    // Open file for reading points
    if ((fp = fopen(fileName, "r")) == 0)
    {
        printf("cannot open file %s for reading\n", fileName);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    // K
    fscanf(fp, "%d", &knm[0]);

    // K
    fscanf(fp, "%d", &knm[1]);

    // Max Iteretion
    fscanf(fp, "%d", &knm[2]);

    int numOfStrings = knm[0] * knm[0];
    int stringLength = 2 * knm[1];
    // Allocate array of points end Read data from the file
    *sub = (char *)calloc(stringLength, sizeof(char));
    strings = (char *)calloc(numOfStrings * stringLength, sizeof(char));
    if (!strings || !sub)
    {
        printf("Problem to allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    fscanf(fp, "%s", *sub);
    for (int i = 0; i < (numOfStrings); i++)
        fscanf(fp, "%s", &strings[i * stringLength]);
    fclose(fp);

    return strings;
}

void destruct(int *knm, int *K, int *N, int *maxIter, int *dim, int *stringLength)
{
    *K = knm[0];
    *N = knm[1];
    *maxIter = knm[2];
    dim[0] = dim[1] = *K;
    *stringLength = 2 * (*N);
}

// a1b2c3 -> 123abc
void reorderString(char *str, int len)
{
    char tmp[len];
    for (int i = 1, j = 0, k = 0; i < len; i += 2, j += 2, k++)
    {
        tmp[k] = str[i];
        tmp[len / 2 + k] = str[j];
    }
    strncpy(str, tmp, len);
}

int main(int argc, char *argv[])
{

    int rank, np;
    int reorder = 1, dim[2], period[] = {1, 1}, src_0, src_1, dest_0, dest_1;
    int K, N, maxIter, knm[3], stringLength, found = 0;
    char *subString, *myString, *strings;

    MPI_Status status;
    MPI_Comm comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0)
    {
        strings = readFromFile(FILE_NAME, knm, &subString);
        destruct(knm, &K, &N, &maxIter, dim, &stringLength);
        if (K * K != np)
        {
            printf("usage: number of procceses (%d) different from K*K (%d)\n", np, K * K);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        MPI_Bcast(knm, 3, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast(knm, 3, MPI_INT, 0, MPI_COMM_WORLD);
        destruct(knm, &K, &N, &maxIter, dim, &stringLength);
        subString = (char *)calloc(stringLength, sizeof(char));
    }

    myString = (char *)calloc(stringLength, sizeof(char));
    if (!subString || !myString)
    {
        printf("PROCCES %d: \tProblem to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    MPI_Bcast(subString, stringLength, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
        MPI_Scatter(strings, stringLength, MPI_CHAR, myString, stringLength, MPI_CHAR, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, MPI_CHAR, myString, stringLength, MPI_CHAR, 0, MPI_COMM_WORLD);

    // create cartesian metrix representation for procceses
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);
    // get neighbours rank
    MPI_Cart_shift(comm, 0, 1, &src_0, &dest_0);
    MPI_Cart_shift(comm, 1, 1, &src_1, &dest_1);

    for (int i = 0; i < maxIter; i++)
    {
        if (strstr(myString, subString))
            found = 1;
        else
            found = 0;
        if (rank == 0)
        {
            int res[np];
            MPI_Gather(&found, 1, MPI_INT, res, 1, MPI_INT, 0, MPI_COMM_WORLD);
            found = 0;
            for (int i = 0; i < np; i++)
                found += res[i];
        }
        else
        {
            MPI_Gather(&found, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (found)
        {
            if (rank == 0)
            {
                MPI_Gather(myString, stringLength, MPI_CHAR, strings, stringLength, MPI_CHAR, 0, MPI_COMM_WORLD);
                for (int i = 0; i < K * K; strings += stringLength, i++)
                    printf("PROCCES %d: \tSTRING: %.*s.\n", i, stringLength, strings);
            }
            else
            {
                MPI_Gather(myString, stringLength, MPI_CHAR, NULL, 0, MPI_CHAR, 0, MPI_COMM_WORLD);
            }
            break;
        }
        reorderString(myString, stringLength);
        MPI_Sendrecv_replace(myString, N, MPI_CHAR, dest_0, 0, src_0, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv_replace(&myString[N], N, MPI_CHAR, dest_1, 0, src_1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank == 0 && !found)
    {
        printf("The string was not found\n");
    }

    MPI_Finalize();

    return 0;
}