/* -*- C -*-
 *
 * $HEADER$
 *
 * The most basic of MPI applications
 */

#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
