/*
    Copyright (C) 2015, The AROS Development Team. All rights reserved.
*/

#include <aros/kernel.h>
#include <aros/libcall.h>

#include <kernel_base.h>
#include "kernel_intern.h"

#include <proto/kernel.h>

AROS_LH2I(int, KrnUnmapGlobal,
        AROS_LHA(void *, virt, A0),
        AROS_LHA(uint32_t, length, D0),
        struct KernelBase *, KernelBase, 17, Kernel)
{
    AROS_LIBFUNC_INIT

    int retval = 0;

    D(bug("[Kernel] KrnUnmapGlobal(%08x, %08x)\n", virt, length));

    return retval;

    AROS_LIBFUNC_EXIT
}
