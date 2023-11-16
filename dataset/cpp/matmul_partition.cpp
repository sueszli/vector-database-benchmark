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

// Maximum Matrix Dimension Supported by Kernel
#define MAX_DIM 16

//TRIPCOUNT identifier
const unsigned int c_dim = MAX_DIM;

extern "C" {
void matmul_partition(const int *in1, // Read-Only Matrix 1
                      const int *in2, // Read-Only Matrix 2
                      int *out_r,     // Output Result
                      int dim) { // Matrix Dimension. Assuming Square Matrix
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out_r offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out_r bundle = control
#pragma HLS INTERFACE s_axilite port = dim bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    int A[MAX_DIM * MAX_DIM];
    int B[MAX_DIM * MAX_DIM];
    int C[MAX_DIM * MAX_DIM];
    //Cyclic Partition for A as matrix multiplication needs row-wise parallel access
#pragma HLS ARRAY_PARTITION variable = A dim = 1 cyclic factor = 16
//Block Partition for B as matrix multiplication needs column-wise parallel access
#pragma HLS ARRAY_PARTITION variable = B dim = 1 block factor = 16

//As A and B Matrix are partitioned with the factor of MAX_DIM, so to get
// parallel row/column access, input square matrix[dimXdim] should be written
// into local Array in MATRIX[MAX_DIM * MAX_DIM] format

// Burst read for matrix A
readA:
    for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
       #pragma HLS PIPELINE II=1
       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
        if (j == dim) {
            j = 0;
            i++;
        }
        A[i * MAX_DIM + j] = in1[itr];
    }

// Burst read for matrix B
readB:
    for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
       #pragma HLS PIPELINE II=1
       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
        if (j == dim) {
            j = 0;
            i++;
        }
        B[i * MAX_DIM + j] = in2[itr];
    }

lreorder1:
    for (int i = 0; i < dim; i++) {
       #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
        //As A and B are partition correctly so loop pipelining is applied
        // at 2nd level loop and which will eventually unroll the lower loop
    lreorder2:
        for (int j = 0; j < dim; j++) {
           #pragma HLS PIPELINE II=1
           #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
            int result = 0;
        lreorder3:
            for (int k = 0; k < MAX_DIM; k++) {
               #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
                result += A[i * MAX_DIM + k] * B[k * MAX_DIM + j];
            }
            C[i * MAX_DIM + j] = result;
        }
    }

// Burst write from output matrices to global memory
// Burst write from matrix C
writeC:
    for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
       #pragma HLS PIPELINE II=1
       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
        if (j == dim) {
            j = 0;
            i++;
        }
        out_r[itr] = C[i * MAX_DIM + j];
    }
}
}
