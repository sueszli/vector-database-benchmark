// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
//
// Copyright 2021 by Geza Lore. This program is free software; you can
// redistribute it and/or modify it under the terms of either the GNU
// Lesser General Public License Version 3 or the Perl Artistic License
// Version 2.0.
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//*************************************************************************

#include <Vt_order_dpi_export_2.h>
#include <Vt_order_dpi_export_2__Dpi.h>
#include <svdpi.h>

void toggle_other_clk(svBit val) { set_other_clk(val); }

int main(int argc, char* argv[]) {
    VM_PREFIX* const tb = new VM_PREFIX;
    tb->contextp()->commandArgs(argc, argv);
    bool clk = true;

    while (!tb->contextp()->gotFinish()) {
        // Timeout
        if (tb->contextp()->time() > 100000) break;
        // Toggle and set main clock
        clk = !clk;
        tb->clk = clk;
        // Eval
        tb->eval();
        // Advance time
        tb->contextp()->timeInc(500);
    }

    delete tb;
    return 0;
}
