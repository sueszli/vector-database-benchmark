/* -*- C -*-
 *
 * $HEADER$
 *
 * The most basic of MPI applications
 */

#include "mpi.h"
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int rank, size;
    char *foo = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello, World, I am %d of %d\n", rank, size);

    if (1 == rank) {
        sleep(2);
        *foo = 42;
    }

    MPI_Finalize();
    return 0;
}
