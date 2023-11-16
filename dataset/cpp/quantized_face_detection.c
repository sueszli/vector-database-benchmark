// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string.h>

#include "quantized_face_detection.h"
#include "quantized_utils.h"
#include "quantized_fastgrnn.h"
#include "quantized_rnnpool.h"
#include "quantized_mbconv.h"

#include "q_scut_head_b_face2_model/conv2D.h"
#include "q_scut_head_b_face2_model/rnn1.h"
#include "q_scut_head_b_face2_model/rnn2.h"
#include "q_scut_head_b_face2_model/mbconv1.h"
#include "q_scut_head_b_face2_model/mbconv2.h"
#include "q_scut_head_b_face2_model/mbconv3.h"
#include "q_scut_head_b_face2_model/mbconv4.h"
#include "q_scut_head_b_face2_model/detection1.h"
#include "q_scut_head_b_face2_model/mbconv5.h"
#include "q_scut_head_b_face2_model/mbconv6.h"
#include "q_scut_head_b_face2_model/mbconv7.h"
#include "q_scut_head_b_face2_model/mbconv8.h"
#include "q_scut_head_b_face2_model/detection2.h"
#include "q_scut_head_b_face2_model/mbconv9.h"
#include "q_scut_head_b_face2_model/mbconv10.h"
#include "q_scut_head_b_face2_model/mbconv11.h"
#include "q_scut_head_b_face2_model/detection3.h"
#include "q_scut_head_b_face2_model/mbconv12.h"
#include "q_scut_head_b_face2_model/mbconv13.h"
#include "q_scut_head_b_face2_model/mbconv14.h"
#include "q_scut_head_b_face2_model/detection4.h"

void q_face_detection(char* const mem_buf) {
  // Conv2D Sub-Pipeline
  q7xq15_q7_convolution((Q7_T*)mem_buf, CBR1F, (Q7_T*)(mem_buf + 76800),
    CONV2D_N, CBR1F_H, CBR1F_W, CBR1F_CIN, CBR1F_HF, CBR1F_WF, CBR1F_CF,
    CONV2D_COUT, CONV2D_HOUT, CONV2D_WOUT, CBR1F_G, CBR1F_HPADL, CBR1F_HPADR,
    CBR1F_WPADL, CBR1F_WPADR, CBR1F_HSTRIDE, CBR1F_WSTRIDE, CBR1F_HDILATION,
    CBR1F_WDILATION, CBR1F_Scinput, CBR1F_Scoutput, CBR1F_Demote);

  q7xq15_q7_t_add_vec((Q7_T*)(mem_buf + 76800), CBR1B, CONV2D_N, CONV2D_HOUT,
    CONV2D_WOUT, CONV2D_COUT, (Q7_T*)mem_buf, CBR1B_Scten, CBR1B_Scvec,
    CBR1B_Scret);

  q7xq15_q7_convolution((Q7_T*)mem_buf, CBR1W, (Q7_T*)mem_buf, CONV2D_N,
    CONV2D_HOUT, CONV2D_WOUT, CONV2D_COUT, CBR1W_HF, CBR1W_WF, CBR1W_CF,
    CBR1W_COUT, CONV2D_HOUT, CONV2D_WOUT, CBR1W_G, CBR1W_HPADL, CBR1W_HPADR,
    CBR1W_WPADL, CBR1W_WPADR, CBR1W_HSTRIDE, CBR1W_WSTRIDE, CBR1W_HDILATION,
    CBR1W_WDILATION, CBR1W_Scinput, CBR1W_Scoutput, CBR1W_Demote);

  q7_t_relu((Q7_T*)mem_buf, CONV2D_N, CONV2D_HOUT, CONV2D_WOUT, CONV2D_COUT,
    (Q7_T*)(mem_buf + 76800), CONV2D_Limit, CONV2D_Div);

  Q7_T* mem_buf_offset_q7 = (Q7_T*)mem_buf;
  Q15_T* mem_buf_offset_q15 = (Q15_T*)mem_buf;

  // RNNPool Sub-Pipeline
  memset(mem_buf, 0, sizeof(Q7_T) * 76800);
  memset((mem_buf + 153600), 0, sizeof(Q15_T));
  memset((mem_buf + 153602), 0, sizeof(Q15_T));

  for (ITER_T patch_x = 0; (patch_x < 29); patch_x++) {
    for (ITER_T patch_y = 0; (patch_y < 39); patch_y++) {
      q7xq15_q15_rnnpool_block((Q7_T*)(mem_buf + 76800 + ((2560 * patch_x) + (16 * patch_y))),
        INPUT_CHANNELS, PATCH_DIM, CONV2D_WOUT, q7xq15_q15_fastgrnn, HIDDEN_DIM1,
        (const void*)(&RNN1_PARAMS), (void*)(&RNN1_BUFFERS),
        (const void*)(&RNN1_SCALES), q15_fastgrnn, HIDDEN_DIM2,
        (const void*)(&RNN2_PARAMS), (void*)(&RNN2_BUFFERS),
        (const void*)(&RNN2_SCALES), (Q15_T*)(mem_buf + 153750),
        (Q15_T*)(mem_buf + 153900), ShR1, ShL1, ShR2, ShL2);

      for (ITER_T i = 0; i < 64; i++) {
        mem_buf_offset_q7[patch_x * 2560 + patch_y * 64 + i] = (Q7_T)(mem_buf_offset_q15[76875 + i]);
      }
    }
  }

  memcpy(&mem_buf_offset_q7[29 * 2560], &mem_buf_offset_q7[28 * 2560],
         39 * 64 * sizeof(Q7_T));
  for (ITER_T i = 0; i < 30; i++) {
    memcpy(&mem_buf_offset_q7[39 * 64 + i * 2560],
           &mem_buf_offset_q7[38 * 64 + i * 2560], 64 * sizeof(Q7_T));
  }

  // MBConv Sub-Pipeline
  // MBConv Layer 1
  q7xq15_q15_mbconv_block((Q7_T*)mem_buf, L1_F1, L1_W1, L1_B1, L1_F2, L1_W2,
    L1_B2, L1_F3, L1_W3, L1_B3, (Q15_T*)(mem_buf + 76800),
    (Q15_T*)(mem_buf + 153600), (Q15_T*)(mem_buf + 184500), L1_N, L1_H, L1_W,
    L1_CIN, L1_CTEMP, L1_HF, L1_WF, L1_COUT, L1_HOUT, L1_WOUT, L1_HPADL,
    L1_HPADR, L1_WPADL, L1_WPADR, L1_HSTRIDE, L1_WSTRIDE, L1_Limit1, L1_Limit2,
    L1_ShRU1, L1_ShRX1, L1_ShRU2, L1_ShRX2, L1_ShRU3, L1_ShRW3, L1_ShLU1,
    L1_ShLX1, L1_ShLU2, L1_ShLX2, L1_ShLU3, L1_ShLW3);

  // MBConv Layer 2
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L2_F1, L2_W1, L2_B1, L2_F2,
    L2_W2, L2_B2, L2_F3, L2_W3, L2_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 153600), (Q15_T*)(mem_buf + 169050), L2_N, L2_H, L2_W,
    L2_CIN, L2_CTEMP, L2_HF, L2_WF, L2_COUT, L2_HOUT, L2_WOUT, L2_HPADL,
    L2_HPADR, L2_WPADL, L2_WPADR, L2_HSTRIDE, L2_WSTRIDE, L2_Limit1, L2_Limit2,
    L2_ShRU1, L2_ShRX1, L2_ShRU2, L2_ShRX2, L2_ShRU3, L2_ShRW3, L2_ShLU1,
    L2_ShLX1, L2_ShLU2, L2_ShLX2, L2_ShLU3, L2_ShLW3);

  // MBConv1 + MBConv2
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L2_N, L2_HOUT, L2_WOUT,
    L2_COUT, (Q15_T*)(mem_buf + 76800), L2_Scten1, L2_Scten2, L2_Scret);

  // MBConv Layer 3
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L3_F1, L3_W1, L3_B1, L3_F2,
    L3_W2, L3_B2, L3_F3, L3_W3, L3_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 153600), (Q15_T*)(mem_buf + 169050), L3_N, L3_H, L3_W,
    L3_CIN, L3_CTEMP, L3_HF, L3_WF, L3_COUT, L3_HOUT, L3_WOUT, L3_HPADL,
    L3_HPADR, L3_WPADL, L3_WPADR, L3_HSTRIDE, L3_WSTRIDE, L3_Limit1, L3_Limit2,
    L3_ShRU1, L3_ShRX1, L3_ShRU2, L3_ShRX2, L3_ShRU3, L3_ShRW3, L3_ShLU1,
    L3_ShLX1, L3_ShLU2, L3_ShLX2, L3_ShLU3, L3_ShLW3);

  // MBConv1 + MBConv2 + MBConv3
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L3_N, L3_HOUT, L3_WOUT,
    L3_COUT, (Q15_T*)(mem_buf + 76800), L3_Scten1, L3_Scten2, L3_Scret);

  // MBConv Layer 4
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L4_F1, L4_W1, L4_B1, L4_F2,
    L4_W2, L4_B2, L4_F3, L4_W3, L4_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 153600), (Q15_T*)(mem_buf + 169050), L4_N, L4_H, L4_W,
    L4_CIN, L4_CTEMP, L4_HF, L4_WF, L4_COUT, L4_HOUT, L4_WOUT, L4_HPADL,
    L4_HPADR, L4_WPADL, L4_WPADR, L4_HSTRIDE, L4_WSTRIDE, L4_Limit1, L4_Limit2,
    L4_ShRU1, L4_ShRX1, L4_ShRU2, L4_ShRX2, L4_ShRU3, L4_ShRW3, L4_ShLU1,
    L4_ShLX1, L4_ShLU2, L4_ShLX2, L4_ShLU3, L4_ShLW3);

  // MBConv1 + MBConv2 + MBConv3 + MBConv4
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L4_N, L4_HOUT, L4_WOUT,
    L4_COUT, (Q15_T*)(mem_buf + 76800), L4_Scten1, L4_Scten2, L4_Scret);

  // Detection Layer 1 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)(mem_buf + 76800), L4_N, L4_HOUT, L4_WOUT, L4_COUT,
    (Q15_T*)mem_buf, D1_ScaleIn, D1_ScaleOut);

  q15_convolution((Q15_T*)mem_buf, D1NW, (Q15_T*)mem_buf, L4_N, L4_HOUT,
    L4_WOUT, L4_COUT, D1NW_HF, D1NW_WF, D1NW_CF, D1NW_COUT, L4_HOUT, L4_WOUT,
    D1NW_G, D1NW_HPADL, D1NW_HPADR, D1NW_WPADL, D1NW_WPADR, D1NW_HSTRIDE,
    D1NW_WSTRIDE, D1NW_HDILATION, D1NW_WDILATION, D1NW_Scinput, D1NW_Scoutput,
    D1NW_Demote);

  q15_convolution((Q15_T*)mem_buf, D1CW, (Q15_T*)(mem_buf + 153600), L4_N,
    L4_HOUT, L4_WOUT, D1NW_COUT * D1NW_G, D1CW_HF, D1CW_WF, D1CW_CF, D1CW_COUT,
    L4_HOUT, L4_WOUT, D1CW_G, D1CW_HPADL, D1CW_HPADR, D1CW_WPADL, D1CW_WPADR,
    D1CW_HSTRIDE, D1CW_WSTRIDE, D1CW_HDILATION, D1CW_WDILATION, D1CW_Scinput,
    D1CW_Scoutput, D1CW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 153600), D1CB, L4_N, L4_HOUT, L4_WOUT,
    D1CW_COUT, (Q15_T*)(mem_buf + 172800), D1CB_Scten, D1CB_Scvec, D1CB_Scret);

  q15_convolution((Q15_T*)mem_buf, D1LW, (Q15_T*)(mem_buf + 153600), L4_N,
    L4_HOUT, L4_WOUT, D1NW_COUT * D1NW_G, D1LW_HF, D1LW_WF, D1LW_CF, D1LW_COUT,
    L4_HOUT, L4_WOUT, D1LW_G, D1LW_HPADL, D1LW_HPADR, D1LW_WPADL, D1LW_WPADR,
    D1LW_HSTRIDE, D1LW_WSTRIDE, D1LW_HDILATION, D1LW_WDILATION, D1LW_Scinput,
    D1LW_Scoutput, D1LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 153600), D1LB, L4_N, L4_HOUT, L4_WOUT,
    D1LW_COUT, (Q15_T*)(mem_buf + 153600), D1LB_Scten, D1LB_Scvec, D1LB_Scret);

  memset((mem_buf_offset_q15 + 81600), 0, sizeof(Q15_T) * 2400);
  memset(mem_buf_offset_q15, 0, sizeof(Q15_T) * 1);
  memset((mem_buf_offset_q15 + 1), 0, sizeof(Q15_T) * 1);

  for (ITER_T i = 0; i < 30; i++) {
    for (ITER_T j = 0; j < 40; j++) {
      for (ITER_T k = 0; k < 3; k++) {
        mem_buf_offset_q15[9 + k] = mem_buf_offset_q15[86400 + (i * 160 + j * 4 + k)];
      }

      ITER_T index;
      q15_v_argmax(&mem_buf_offset_q15[9], 3, &index);

      mem_buf_offset_q15[81600 + (i * 80 + j * 2)] = mem_buf_offset_q15[86400 + (i * 160 + j * 4 + index)];
      mem_buf_offset_q15[81600 + (i * 80 + j * 2 + 1)] = mem_buf_offset_q15[86400 + (i * 160 + j * 4 + 3)];
    }
  }

  // MBConv Layer 5
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L5_F1, L5_W1, L5_B1, L5_F2,
    L5_W2, L5_B2, L5_F3, L5_W3, L5_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 172800), (Q15_T*)(mem_buf + 168300), L5_N, L5_H, L5_W,
    L5_CIN, L5_CTEMP, L5_HF, L5_WF, L5_COUT, L5_HOUT, L5_WOUT, L5_HPADL,
    L5_HPADR, L5_WPADL, L5_WPADR, L5_HSTRIDE, L5_WSTRIDE, L5_Limit1, L5_Limit2,
    L5_ShRU1, L5_ShRX1, L5_ShRU2, L5_ShRX2, L5_ShRU3, L5_ShRW3, L5_ShLU1,
    L5_ShLX1, L5_ShLU2, L5_ShLX2, L5_ShLU3, L5_ShLW3);

  // MBConv1 + MBConv2 + MBConv3 + MBConv4 + MBConv5
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L5_N, L5_HOUT, L5_WOUT,
    L5_COUT, (Q15_T*)(mem_buf + 76800), L5_Scten1, L5_Scten2, L5_Scret);

  // MBConv Layer 6
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L6_F1, L6_W1, L6_B1, L6_F2,
    L6_W2, L6_B2, L6_F3, L6_W3, L6_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 172800), (Q15_T*)(mem_buf + 168300), L6_N, L6_H, L6_W,
    L6_CIN, L6_CTEMP, L6_HF, L6_WF, L6_COUT, L6_HOUT, L6_WOUT, L6_HPADL,
    L6_HPADR, L6_WPADL, L6_WPADR, L6_HSTRIDE, L6_WSTRIDE, L6_Limit1, L6_Limit2,
    L6_ShRU1, L6_ShRX1, L6_ShRU2, L6_ShRX2, L6_ShRU3, L6_ShRW3, L6_ShLU1,
    L6_ShLX1, L6_ShLU2, L6_ShLX2, L6_ShLU3, L6_ShLW3);

  // MBConv1 + MBConv2 + MBConv3 + MBConv4 + MBConv5 + MBConv6
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L6_N, L6_HOUT, L6_WOUT,
    L6_COUT, (Q15_T*)(mem_buf + 76800), L6_Scten1, L6_Scten2, L6_Scret);

  // MBConv Layer 7
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L7_F1, L7_W1, L7_B1, L7_F2,
    L7_W2, L7_B2, L7_F3, L7_W3, L7_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 172800), (Q15_T*)(mem_buf + 168300), L7_N, L7_H, L7_W,
    L7_CIN, L7_CTEMP, L7_HF, L7_WF, L7_COUT, L7_HOUT, L7_WOUT, L7_HPADL,
    L7_HPADR, L7_WPADL, L7_WPADR, L7_HSTRIDE, L7_WSTRIDE, L7_Limit1, L7_Limit2,
    L7_ShRU1, L7_ShRX1, L7_ShRU2, L7_ShRX2, L7_ShRU3, L7_ShRW3, L7_ShLU1,
    L7_ShLX1, L7_ShLU2, L7_ShLX2, L7_ShLU3, L7_ShLW3);

  // MBConv1 + MBConv2 + MBConv3 + MBConv4 + MBConv5 + MBConv6 + MBConv7
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L7_N, L7_HOUT, L7_WOUT,
    L7_COUT, (Q15_T*)(mem_buf + 76800), L7_Scten1, L7_Scten2, L7_Scret);

  // MBConv Layer 8
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L8_F1, L8_W1, L8_B1, L8_F2,
    L8_W2, L8_B2, L8_F3, L8_W3, L8_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 172800), (Q15_T*)(mem_buf + 168300), L8_N, L8_H, L8_W,
    L8_CIN, L8_CTEMP, L8_HF, L8_WF, L8_COUT, L8_HOUT, L8_WOUT, L8_HPADL,
    L8_HPADR, L8_WPADL, L8_WPADR, L8_HSTRIDE, L8_WSTRIDE, L8_Limit1, L8_Limit2,
    L8_ShRU1, L8_ShRX1, L8_ShRU2, L8_ShRX2, L8_ShRU3, L8_ShRW3, L8_ShLU1,
    L8_ShLX1, L8_ShLU2, L8_ShLX2, L8_ShLU3, L8_ShLW3);

  // MBConv1 + MBConv2 + MBConv3 + MBConv4 + MBConv5 + MBConv6 + MBConv7 + MBConv8
  q15_t_add((Q15_T*)(mem_buf + 76800), (Q15_T*)mem_buf, L8_N, L8_HOUT, L8_WOUT,
    L8_COUT, (Q15_T*)(mem_buf + 76800), L8_Scten1, L8_Scten2, L8_Scret);

  // Detection Layer 2 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)(mem_buf + 76800), L8_N, L8_HOUT, L8_WOUT, L8_COUT,
    (Q15_T*)mem_buf, D2_ScaleIn, D2_ScaleOut);

  q15_convolution((Q15_T*)mem_buf, D2NW, (Q15_T*)mem_buf, L8_N, L8_HOUT,
    L8_WOUT, L8_COUT, D2NW_HF, D2NW_WF, D2NW_CF, D2NW_COUT, L8_HOUT, L8_WOUT,
    D2NW_G, D2NW_HPADL, D2NW_HPADR, D2NW_WPADL, D2NW_WPADR, D2NW_HSTRIDE,
    D2NW_WSTRIDE, D2NW_HDILATION, D2NW_WDILATION, D2NW_Scinput, D2NW_Scoutput,
    D2NW_Demote);

  q15_convolution((Q15_T*)mem_buf, D2CW, (Q15_T*)(mem_buf + 168000), L8_N,
    L8_HOUT, L8_WOUT, D2NW_COUT * D2NW_G, D2CW_HF, D2CW_WF, D2CW_CF, D2CW_COUT,
    L8_HOUT, L8_WOUT, D2CW_G, D2CW_HPADL, D2CW_HPADR, D2CW_WPADL, D2CW_WPADR,
    D2CW_HSTRIDE, D2CW_WSTRIDE, D2CW_HDILATION, D2CW_WDILATION, D2CW_Scinput,
    D2CW_Scoutput, D2CW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 168000), D2CB, L8_N, L8_HOUT, L8_WOUT,
    D2CW_COUT, (Q15_T*)(mem_buf + 168000), D2CB_Scten, D2CB_Scvec, D2CB_Scret);

  q15_convolution((Q15_T*)mem_buf, D2LW, (Q15_T*)(mem_buf + 172800), L8_N,
    L8_HOUT, L8_WOUT, D2NW_COUT * D2NW_G, D2LW_HF, D2LW_WF, D2LW_CF, D2LW_COUT,
    L8_HOUT, L8_WOUT, D2LW_G, D2LW_HPADL, D2LW_HPADR, D2LW_WPADL, D2LW_WPADR,
    D2LW_HSTRIDE, D2LW_WSTRIDE, D2LW_HDILATION, D2LW_WDILATION, D2LW_Scinput,
    D2LW_Scoutput, D2LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 172800), D2LB, L8_N, L8_HOUT, L8_WOUT,
    D2LW_COUT, (Q15_T*)(mem_buf + 172800), D2LB_Scten, D2LB_Scvec, D2LB_Scret);

  // MBConv Layer 9
  q15_mbconv_block((Q15_T*)(mem_buf + 76800), L9_F1, L9_W1, L9_B1, L9_F2,
    L9_W2, L9_B2, L9_F3, L9_W3, L9_B3, (Q15_T*)mem_buf,
    (Q15_T*)(mem_buf + 57600), (Q15_T*)(mem_buf + 73050), L9_N, L9_H, L9_W,
    L9_CIN, L9_CTEMP, L9_HF, L9_WF, L9_COUT, L9_HOUT, L9_WOUT, L9_HPADL,
    L9_HPADR, L9_WPADL, L9_WPADR, L9_HSTRIDE, L9_WSTRIDE, L9_Limit1, L9_Limit2,
    L9_ShRU1, L9_ShRX1, L9_ShRU2, L9_ShRX2, L9_ShRU3, L9_ShRW3, L9_ShLU1,
    L9_ShLX1, L9_ShLU2, L9_ShLX2, L9_ShLU3, L9_ShLW3);

  // MBConv Layer 10
  q15xq7_q15_mbconv_block((Q15_T*)mem_buf, L10_F1, L10_W1, L10_B1, L10_F2,
    L10_W2, L10_B2, L10_F3, L10_W3, L10_B3, (Q15_T*)(mem_buf + 96000),
    (Q15_T*)(mem_buf + 72961), (Q15_T*)(mem_buf + 58800), L10_N, L10_H, L10_W,
    L10_CIN, L10_CTEMP, L10_HF, L10_WF, L10_COUT, L10_HOUT, L10_WOUT,
    L10_HPADL, L10_HPADR, L10_WPADL, L10_WPADR, L10_HSTRIDE, L10_WSTRIDE,
    L10_Limit1, L10_Limit2, L10_ShRU1, L10_ShRX1, L10_ShRU2, L10_ShRX2,
    L10_ShRU3, L10_ShRW3, L10_ShLU1, L10_ShLX1, L10_ShLU2, L10_ShLX2,
    L10_ShLU3, L10_ShLW3);

  // MBConv9 + MBConv10
  q15_t_add((Q15_T*)mem_buf, (Q15_T*)(mem_buf + 96000), L10_N, L10_HOUT,
    L10_WOUT, L10_COUT, (Q15_T*)mem_buf, L10_Scten1, L10_Scten2, L10_Scret);

  // MBConv Layer 11
  q15xq7_q15_mbconv_block((Q15_T*)mem_buf, L11_F1, L11_W1, L11_B1, L11_F2,
    L11_W2, L11_B2, L11_F3, L11_W3, L11_B3, (Q15_T*)(mem_buf + 96000),
    (Q15_T*)(mem_buf + 72961), (Q15_T*)(mem_buf + 58800), L11_N, L11_H, L11_W,
    L11_CIN, L11_CTEMP, L11_HF, L11_WF, L11_COUT, L11_HOUT, L11_WOUT,
    L11_HPADL, L11_HPADR, L11_WPADL, L11_WPADR, L11_HSTRIDE, L11_WSTRIDE,
    L11_Limit1, L11_Limit2, L11_ShRU1, L11_ShRX1, L11_ShRU2, L11_ShRX2,
    L11_ShRU3, L11_ShRW3, L11_ShLU1, L11_ShLX1, L11_ShLU2, L11_ShLX2,
    L11_ShLU3, L11_ShLW3);

  // MBConv9 + MBConv10 + MBConv11
  q15_t_add((Q15_T*)mem_buf, (Q15_T*)(mem_buf + 96000), L11_N, L11_HOUT,
    L11_WOUT, L11_COUT, (Q15_T*)mem_buf, L11_Scten1, L11_Scten2, L11_Scret);

  // Detection Layer 3 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)mem_buf, L11_N, L11_HOUT, L11_WOUT, L11_COUT,
    (Q15_T*)(mem_buf + 96000), D3_ScaleIn, D3_ScaleOut);

  q15_convolution((Q15_T*)(mem_buf + 96000), D3NW, (Q15_T*)(mem_buf + 96000),
    L11_N, L11_HOUT, L11_WOUT, L11_COUT, D3NW_HF, D3NW_WF, D3NW_CF, D3NW_COUT,
    L11_HOUT, L11_WOUT, D3NW_G, D3NW_HPADL, D3NW_HPADR, D3NW_WPADL, D3NW_WPADR,
    D3NW_HSTRIDE, D3NW_WSTRIDE, D3NW_HDILATION, D3NW_WDILATION, D3NW_Scinput,
    D3NW_Scoutput, D3NW_Demote);

  q15_convolution((Q15_T*)(mem_buf + 96000), D3CW, (Q15_T*)(mem_buf + 61200),
    L11_N, L11_HOUT, L11_WOUT, D3NW_COUT * D3NW_G, D3CW_HF, D3CW_WF, D3CW_CF,
    D3CW_COUT, L11_HOUT, L11_WOUT, D3CW_G, D3CW_HPADL, D3CW_HPADR, D3CW_WPADL,
    D3CW_WPADR, D3CW_HSTRIDE, D3CW_WSTRIDE, D3CW_HDILATION, D3CW_WDILATION,
    D3CW_Scinput, D3CW_Scoutput, D3CW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 61200), D3CB, L11_N, L11_HOUT, L11_WOUT,
    D3CW_COUT, (Q15_T*)(mem_buf + 57600), D3CB_Scten, D3CB_Scvec, D3CB_Scret);

  q15_convolution((Q15_T*)(mem_buf + 96000), D3LW, (Q15_T*)(mem_buf + 60000),
    L11_N, L11_HOUT, L11_WOUT, D3NW_COUT * D3NW_G, D3LW_HF, D3LW_WF, D3LW_CF,
    D3LW_COUT, L11_HOUT, L11_WOUT, D3LW_G, D3LW_HPADL, D3LW_HPADR, D3LW_WPADL,
    D3LW_WPADR, D3LW_HSTRIDE, D3LW_WSTRIDE, D3LW_HDILATION, D3LW_WDILATION,
    D3LW_Scinput, D3LW_Scoutput, D3LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 60000), D3LB, L11_N, L11_HOUT, L11_WOUT,
    D3LW_COUT, (Q15_T*)(mem_buf + 60000), D3LB_Scten, D3LB_Scvec, D3LB_Scret);

  // MBConv Layer 12
  q15xq7_q7_mbconv_block((Q15_T*)mem_buf, L12_F1, L12_W1, L12_B1, L12_F2,
    L12_W2, L12_B2, L12_F3, L12_W3, L12_B3, (Q7_T*)(mem_buf + 76800),
    (Q15_T*)(mem_buf + 115200), (Q15_T*)(mem_buf + 62400), L12_N, L12_H, L12_W,
    L12_CIN, L12_CTEMP, L12_HF, L12_WF, L12_COUT, L12_HOUT, L12_WOUT,
    L12_HPADL, L12_HPADR, L12_WPADL, L12_WPADR, L12_HSTRIDE, L12_WSTRIDE,
    L12_Limit1, L12_Limit2, L12_ShRU1, L12_ShRX1, L12_ShRU2, L12_ShRX2,
    L12_ShRU3, L12_ShRW3, L12_ShLU1, L12_ShLX1, L12_ShLU2, L12_ShLX2,
    L12_ShLU3, L12_ShLW3);

  // MBConv Layer 13
  q7_mbconv_block((Q7_T*)(mem_buf + 76800), L13_F1, L13_W1, L13_B1, L13_F2,
    L13_W2, L13_B2, L13_F3, L13_W3, L13_B3, (Q7_T*)mem_buf,
    (Q7_T*)(mem_buf + 38400), (Q7_T*)(mem_buf + 54600), L13_N, L13_H, L13_W,
    L13_CIN, L13_CTEMP, L13_HF, L13_WF, L13_COUT, L13_HOUT, L13_WOUT,
    L13_HPADL, L13_HPADR, L13_WPADL, L13_WPADR, L13_HSTRIDE, L13_WSTRIDE,
    L13_Limit1, L13_Limit2, L13_ShRU1, L13_ShRX1, L13_ShRU2, L13_ShRX2,
    L13_ShRU3, L13_ShRW3, L13_ShLU1, L13_ShLX1, L13_ShLU2, L13_ShLX2,
    L13_ShLU3, L13_ShLW3);

  // MBConv12 + MBConv13
  q7_t_add((Q7_T*)(mem_buf + 76800), (Q7_T*)mem_buf, L13_N, L13_HOUT, L13_WOUT,
    L13_COUT, (Q7_T*)(mem_buf + 76800), L13_Scten1, L13_Scten2, L13_Scret);

  // MBConv Layer 14
  q7_mbconv_block((Q7_T*)(mem_buf + 76800), L14_F1, L14_W1, L14_B1, L14_F2,
    L14_W2, L14_B2, L14_F3, L14_W3, L14_B3, (Q7_T*)mem_buf,
    (Q7_T*)(mem_buf + 38400), (Q7_T*)(mem_buf + 54600), L14_N, L14_H, L14_W,
    L14_CIN, L14_CTEMP, L14_HF, L14_WF, L14_COUT, L14_HOUT, L14_WOUT,
    L14_HPADL, L14_HPADR, L14_WPADL, L14_WPADR, L14_HSTRIDE, L14_WSTRIDE,
    L14_Limit1, L14_Limit2, L14_ShRU1, L14_ShRX1, L14_ShRU2, L14_ShRX2,
    L14_ShRU3, L14_ShRW3, L14_ShLU1, L14_ShLX1, L14_ShLU2, L14_ShLX2,
    L14_ShLU3, L14_ShLW3);

  // MBConv12 + MBConv13 + MBConv14
  q7_t_add((Q7_T*)(mem_buf + 76800), (Q7_T*)mem_buf, L14_N, L14_HOUT, L14_WOUT,
    L14_COUT, (Q7_T*)(mem_buf + 76800), L14_Scten1, L14_Scten2, L14_Scret);

  // Detection Layer 4 Sub-Pipeline
  q7xq15_q7_convolution((Q7_T*)(mem_buf + 76800), D4CW,
    (Q7_T*)(mem_buf + 4800), L14_N, L14_HOUT, L14_WOUT, L14_COUT, D4CW_HF,
    D4CW_WF, D4CW_CF, D4CW_COUT, L14_HOUT, L14_WOUT, D4CW_G, D4CW_HPADL,
    D4CW_HPADR, D4CW_WPADL, D4CW_WPADR, D4CW_HSTRIDE, D4CW_WSTRIDE,
    D4CW_HDILATION, D4CW_WDILATION, D4CW_Scinput,
    D4CW_Scoutput, D4CW_Demote);

  q7xq15_q7_t_add_vec((Q7_T*)(mem_buf + 4800), D4CB, L14_N, L14_HOUT,
    L14_WOUT, D4CW_COUT, (Q7_T*)mem_buf, D4CB_Scten, D4CB_Scvec, D4CB_Scret);

  q7xq15_q15_convolution((Q7_T*)(mem_buf + 76800), D4LW,
    (Q15_T*)(mem_buf + 2400), L14_N, L14_HOUT, L14_WOUT, L14_COUT, D4LW_HF,
    D4LW_WF, D4LW_CF, D4LW_COUT, L14_HOUT, L14_WOUT, D4LW_G, D4LW_HPADL,
    D4LW_HPADR, D4LW_WPADL, D4LW_WPADR, D4LW_HSTRIDE, D4LW_WSTRIDE,
    D4LW_HDILATION, D4LW_WDILATION, D4LW_Scinput, D4LW_Scoutput, D4LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 2400), D4LB, L14_N, L14_HOUT, L14_WOUT, D4LW_COUT,
    (Q15_T*)(mem_buf + 2400), D4LB_Scten, D4LB_Scvec, D4LB_Scret);

  // Re-ordering the outputs
  memset((mem_buf_offset_q15 + 38400), 0, sizeof(Q15_T) * 18000);
  memcpy(&mem_buf_offset_q15[38400], &mem_buf_offset_q15[81600], 2400 * sizeof(Q15_T));

  for (ITER_T i = 0; i < 2400; i++) {
    mem_buf_offset_q15[38400 + (i + 2400)] = (mem_buf_offset_q15[84000 + i] / 2);
  }

  for (ITER_T i = 0; i < 600; i++) {
    mem_buf_offset_q15[38400 + (i + 4800)] = (mem_buf_offset_q15[28800 + i] / 2);
  }

  for (ITER_T i = 0; i < 600; i++) {
    mem_buf_offset_q15[38400 + (i + 5400)] = (((Q15_T)mem_buf_offset_q7[i]) << 7);
  }

  memcpy(&mem_buf_offset_q15[38400 + 6000], &mem_buf_offset_q15[76800], 4800 * sizeof(Q15_T));
  memcpy(&mem_buf_offset_q15[38400 + 10800], &mem_buf_offset_q15[86400], 4800 * sizeof(Q15_T));
  memcpy(&mem_buf_offset_q15[38400 + 15600], &mem_buf_offset_q15[30000], 1200 * sizeof(Q15_T));

  for (ITER_T i = 0; (i < 1200); i++) {
    mem_buf_offset_q15[38400 + (i + 16800)] = (mem_buf_offset_q15[1200 + i] / 2);
  }
}
