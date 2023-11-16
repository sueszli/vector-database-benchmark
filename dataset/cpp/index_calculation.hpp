/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <embedding/common.hpp>

namespace embedding {
using core::CoreResourceManager;

struct IndexCalculationTempStorage {
  core23::Tensor flag;
  core23::Tensor temp_select_storage;
  core23::Tensor temp_scan_storage;

  void init(const std::shared_ptr<CoreResourceManager> &core, int max_num_keys_before_filter,
            int max_num_keys_after_filter, int batch_size_before_filter,
            int batch_size_after_filter, int num_lookup);
};

template <typename KeySelector>
class IndexCalculation {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  KeySelector key_selector_;
  IndexCalculationTempStorage temp_storage_;

 public:
  void init(std::shared_ptr<CoreResourceManager> core, const KeySelector &key_selector,
            int batch_size);

  void filter_sparse_input(const core23::Tensor &keys, const core23::Tensor &bucket_range,
                           EmbeddingInput &result, int batch_size);
};

struct ReductionIndices {
  core23::Tensor sorted_keys;
  core23::Tensor src_ids;
  core23::Tensor dst_ids;
  core23::Tensor table_ids;
  core23::Tensor ev_sizes;
  core23::Tensor num_key;

  size_t num_elements;

  void init(std::shared_ptr<CoreResourceManager> core, int local_hotness_sum, int batch_size,
            core23::DataType key_type);
};

struct DenseReductionIndices {
  const core23::Tensor *model_reverse_idx;
  int ev_size;
  size_t reverse_key_num;
  size_t num_valid_dst_tensor;
};

struct PartitionedResult {
  core23::Tensor partitioned_keys;
  core23::Tensor partitioned_src_ids;
  core23::Tensor partitioned_bucket_range;

  PartitionedResult() = default;

  PartitionedResult(std::shared_ptr<CoreResourceManager> core, int num_lookup,
                    int local_hotness_sum, int batch_size, core23::DataType key_type,
                    core23::DataType offset_type);
};

struct LocalReduceIndexCalculationTempStorage {
  core23::Tensor temp_scan_storage;

  template <typename offset_t>
  void init(const std::shared_ptr<CoreResourceManager> &core, int num_lookup, int batch_size);
};

struct SortInput {
  // for sort
  core23::Tensor keys;
  core23::Tensor src_ids;
  size_t h_num_key;
  core23::Tensor bucket_range;
};

struct SortOutput {
  core23::Tensor sorted_keys;
  core23::Tensor sorted_src_ids;
};

using SortKeyAndSrcIdOp =
    std::function<void(SortInput &, SortOutput &, std::shared_ptr<CoreResourceManager> core)>;

struct SegmentedSortDevice {
 public:
  SegmentedSortDevice() = default;

  SegmentedSortDevice(const std::shared_ptr<CoreResourceManager> &core,
                      core23::Tensor sorted_table_ids, int max_num_keys, int batch_size,
                      int num_lookup, int num_table, core23::DataType key_type);

  void operator()(SortInput &input, SortOutput &output, std::shared_ptr<CoreResourceManager> core);

 private:
  size_t max_key_num_;
  size_t cub_sort_temp_bytes_ = 0;
  core23::Tensor cub_sort_temp_buffer_;  // Void

  core23::Tensor temp_select_storage;
  core23::Tensor d_num_selected_table_range_;
  core23::Tensor temp_lookup_range;

  core23::Tensor partitioned_table_range;

  core23::Tensor sorted_table_ids_;
  int num_lookup_;
  int num_table_;
  int batch_size_;
};

struct IndicesSort {
  core23::Tensor d_temp_sort_storage;

  IndicesSort() = default;

  IndicesSort(const std::shared_ptr<CoreResourceManager> &core, int max_num_keys, int batch_size,
              core23::DataType key_type);

  void operator()(SortInput &input, SortOutput &output, std::shared_ptr<CoreResourceManager> core);
};

struct SegmentdUnique {
  // SegmentdUnique need separate in 3 steps:
  // 1. record 2 buffer, first buffer is record key first appear
  //    second buffer is record table first appear
  // 2. scan step 1 first buffer
  // 3. put the all first buffer compact in output buffer
  //    put table offset in final buffer

  size_t max_key_num_;
  size_t cub_scan_temp_bytes_ = 0;
  core23::Tensor cub_scan_temp_buffer_;  // Void
  core23::Tensor key_flag_buffer_;       // size_t

  SegmentdUnique() = default;

  SegmentdUnique(const std::shared_ptr<CoreResourceManager> &core, int max_num_keys,
                 int batch_size);

  void operator()(const core23::Tensor &sorted_keys, const core23::Tensor &table_ids,
                  const core23::Tensor &key_num, core23::Tensor &unique_keys,
                  core23::Tensor &unique_table_ids, core23::Tensor &unique_keys_offset,
                  core23::Tensor &num_unique_keys, core23::Tensor &dst_ids, size_t h_num_key,
                  bool is_same_ev_size, int ev_size, std::shared_ptr<CoreResourceManager> core);
};

struct CalDstIds {
  size_t max_key_num_;
  size_t cub_scan_temp_bytes_ = 0;
  core23::Tensor cub_scan_temp_buffer_;  // Void

  CalDstIds() = default;

  CalDstIds(const std::shared_ptr<CoreResourceManager> &core, int max_num_keys, int batch_size);

  void operator()(core23::Tensor &sorted_keys, int num_table, const core23::Tensor &table_range,
                  core23::Tensor &dst_ids, std::shared_ptr<CoreResourceManager> core,
                  cudaStream_t stream);
};

struct CalDstOffsetMP {
  size_t max_key_num_;
  size_t cub_scan_temp_bytes_ = 0;
  core23::Tensor cub_scan_temp_buffer_;  // Void

  CalDstOffsetMP() = default;

  CalDstOffsetMP(const std::shared_ptr<CoreResourceManager> &core, int max_num_keys,
                 int batch_size);

  void operator()(const core23::Tensor &unique_key_table_ids,
                  const core23::Tensor &table_id_to_evsizes, const core23::Tensor &num_unique_key,
                  core23::Tensor &dst_key_offset, std::shared_ptr<CoreResourceManager> core,
                  cudaStream_t stream);
};

class LocalReduceIndexCalculation {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  PartitionedResult partitioned_result_;
  LocalReduceIndexCalculationTempStorage temp_storage_;

 public:
  LocalReduceIndexCalculation() = default;

  LocalReduceIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_lookup,
                              int local_hotness_sum, int batch_size, core23::DataType key_type,
                              core23::DataType offset_type);

  void cal_for_sparse_input(const EmbeddingInput &embedding_input,
                            SortKeyAndSrcIdOp sort_key_and_src_id_op,
                            SegmentdUnique &segmented_unique, ReductionIndices &reduction_indices,
                            Wgrad &wgrad, int batch_size);
};
}  // namespace embedding
