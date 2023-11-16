/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*
   Loop Fusion

   Loop fusion is a technique to combine a nested loop with its parent. This
   technique produces more efficient hardware in some cases.
  */

#define MAX_DIMS 5
#include <limits.h>
#include <stdlib.h>

// This is a simple implementation of a linear search algorithm. We use two
// loops. The outer loop cycles through each of the search space and the inner
// loop calculates the distance to a particular point.
extern "C" {
void nearest_neighbor(int *out_r,
                      const int *points,
                      const int *search_point,
                      const int len,
                      const int dim) {
#pragma HLS INTERFACE m_axi port = out_r offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = points offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = search_point offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = out_r bundle = control
#pragma HLS INTERFACE s_axilite port = points bundle = control
#pragma HLS INTERFACE s_axilite port = search_point bundle = control
#pragma HLS INTERFACE s_axilite port = len bundle = control
#pragma HLS INTERFACE s_axilite port = dim bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    int best_i = 0;
    int best_dist = INT_MAX;
    int s_point[MAX_DIMS];

read:
    for (int d = 0; d < dim; ++d) {
       #pragma HLS PIPELINE II=1
        s_point[d] = search_point[d];
    }

find_best:
    for (int p = 0; p < len; ++p) {
        int dist = 0;

    // Calculate the distance in a n-dimensional space
    dist_calc:
        for (int c = 0; c < dim; ++c) {
           #pragma HLS PIPELINE II=1
            int dx = points[dim * p + c] - s_point[c];
            dist += dx * dx;
        }

        if (dist < best_dist) {
            best_i = p;
            best_dist = dist;
        }
    }

write_best:
    for (int c = 0; c < dim; ++c) {
       #pragma HLS PIPELINE II=1
        out_r[c] = points[best_i * dim + c];
    }
}
}
