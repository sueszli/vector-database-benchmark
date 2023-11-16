/* ----------------------------------------------------------------------
 * Project: Tiny Training Engine, MCUNetV3
 * Title:   transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *  - Chuang Gan, ganchuang@csail.mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (5U)
#define DIM_KER_Y (5U)
#define STRIDE (2U)
#define IN_PAD (2U)
#define OUT_PAD (1U)

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_IOHW_int8weight_partialCH(float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region
  for (i = 0; i < input_width * 2 + 4; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: Pad the size of (input_height * 2) * (input_width * 2 + 2)
  for (i = 0; i < input_height; i++) {
    // First type of middle
    *two_column_buffer++ = pad_value;
    for (j = 0; j < input_width; j++) {
      *two_column_buffer = pad_value;
      two_column_buffer += 2;
    }
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;

    // Second type of middle
    for (j = 0; j < input_width * 2 + 4; j++) {
      *two_column_buffer++ = pad_value;
    }
  }
  // Bottom region
  for (i = 0; i < input_width * 2 + 4; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }


  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;
  const int8_t* ksrc;
  float ksrc_transposed[25];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width * 2 + 4) * 2;
    src = input_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 2;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer = *src;
        two_column_buffer += 2;
        src += input_depth;
      }

      two_column_buffer += input_width * 2 + 6;
    }

    // Transpose filter data
    if (c < first_k_channel) {
      ksrc = filter_sram++;
    }
    else {
      ksrc = filter_flash++;
    }
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[24 - i] = (float)*ksrc;

      if (c < first_k_channel) {
        ksrc += first_k_channel;
      }
      else {
        ksrc += input_depth - first_k_channel;
      }
    }

    float* out = output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        transpose_depthwise_mac_kernel5_2row_fp_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + j) * output_depth] = TN_MIN(TN_MAX(sum_0, output_activation_min), output_activation_max);
        out[(i * output_width + j + 1) * output_depth] = TN_MIN(TN_MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        transpose_depthwise_mac_kernel5_1row_fp_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + output_width - 1) * output_depth] = TN_MIN(TN_MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 4;
    }

    bias_data++;
    input_data++;
    output_data++;
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
} 

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_IOHW_int8weight(float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region
  for (i = 0; i < input_width * 2 + 4; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: Pad the size of (input_height * 2) * (input_width * 2 + 2)
  for (i = 0; i < input_height; i++) {
    // First type of middle
    *two_column_buffer++ = pad_value;
    for (j = 0; j < input_width; j++) {
      *two_column_buffer = pad_value;
      two_column_buffer += 2;
    }
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;

    // Second type of middle
    for (j = 0; j < input_width * 2 + 4; j++) {
      *two_column_buffer++ = pad_value;
    }
  }
  // Bottom region
  for (i = 0; i < input_width * 2 + 4; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }


  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;
  const int8_t* ksrc;
  float ksrc_transposed[25];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width * 2 + 4) * 2;
    src = input_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 2;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer = *src;
        two_column_buffer += 2;
        src += input_depth;
      }

      two_column_buffer += input_width * 2 + 6;
    }

    // Transpose filter data
    ksrc = filter_data++;
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[24 - i] = (float)*ksrc;
      ksrc += input_depth;
    }

    float* out = output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        transpose_depthwise_mac_kernel5_2row_fp_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + j) * output_depth] = TN_MIN(TN_MAX(sum_0, output_activation_min), output_activation_max);
        out[(i * output_width + j + 1) * output_depth] = TN_MIN(TN_MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        transpose_depthwise_mac_kernel5_1row_fp_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + output_width - 1) * output_depth] = TN_MIN(TN_MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 4;
    }

    bias_data++;
    input_data++;
    output_data++;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
} 
