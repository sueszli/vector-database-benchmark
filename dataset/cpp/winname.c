/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpitestconf.h"
#include "mpitest.h"
#ifdef HAVE_STRING_H
#include <string.h>
#endif

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Win win;
    int cnt, namelen;
    char name[MPI_MAX_OBJECT_NAME], nameout[MPI_MAX_OBJECT_NAME];

    MTest_Init(&argc, &argv);

#if MTEST_HAVE_MIN_MPI_VERSION(4,1)
    int rlen;
    nameout[0] = 0;
    MPI_Win_get_name(MPI_WIN_NULL, nameout, &rlen);
    if (strcmp(nameout, "MPI_WIN_NULL")) {
        errs++;
        printf("Name of win null is %s, should be MPI_WIN_NULL\n", nameout);
    }
#endif

    cnt = 0;
    while (MTestGetWin(&win, 1)) {
        if (win == MPI_WIN_NULL)
            continue;

        sprintf(name, "win-%d", cnt);
        cnt++;
        MPI_Win_set_name(win, name);
        nameout[0] = 0;
        MPI_Win_get_name(win, nameout, &namelen);
        if (strcmp(name, nameout)) {
            errs++;
            printf("Unexpected name, was %s but should be %s\n", nameout, name);
        }

        MTestFreeWin(&win);
    }

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
