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

#include <core23/registry.hpp>
#include <embedding/embedding_table.hpp>

namespace embedding {

class WeightedModelBackward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_local_embedding_;
  int num_sms_;
  int max_ev_size_;

  core23::Tensor grad_ev_;
  core23::Tensor partial_grad_ev_;
  core23::Tensor partial_key_;
  core23::Tensor partial_ev_length_;
  core23::Tensor partial_dst_offset_array_;

 public:
  WeightedModelBackward() = default;

  WeightedModelBackward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                        int num_local_embedding, const std::vector<int> &h_local_hotness_list,
                        const std::vector<int> &h_local_ev_size_list, int universal_batch_size,
                        int max_ev_size, int num_sms);

  void compute(const core23::Tensor &model_comm_buffer,
               const core23::Tensor &unique_key_ev_size_offset,
               const core23::Tensor &unique_key_bucket_idx,
               const core23::Tensor &unique_key_bucket_idx_offset, uint64_t num_unique_key,
               const core23::Tensor &corrdinate_key, const core23::Tensor &coordinate_wgrad_dst_idx,
               const core23::Tensor &d_local_ev_size_offset, int batch_size, int max_ev_size,
               size_t num_model_key, core23::Tensor *grad_ev,
               const core23::Tensor &coordinate_sp_weight);
};
}  // namespace embedding
