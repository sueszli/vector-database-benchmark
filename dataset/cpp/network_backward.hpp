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

namespace embedding {
using core::CoreResourceManager;

struct NetworkIndices;
struct NetworkBuffer;
struct DenseNetworkIndices;
struct DenseNetworkBuffer;

class NetworkBackward {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  NetworkBackward() = default;

  NetworkBackward(std::shared_ptr<CoreResourceManager> core) : core_(core) {}

  void sparse_backward(const core23::Tensor &dp_num_keys_per_bucket,
                       const EmbeddingOutput &top_grad, const NetworkIndices &network_indices,
                       NetworkBuffer &network_buffer, int batch_size);

  void compute(const core23::Tensor &row_lengths, const core23::Tensor &d_combiner_list,
               const core23::Tensor &top_grad, const core23::Tensor &network_ids,
               const core23::Tensor &network_gpu_ids, const core23::Tensor &network_offsets,
               const core23::Tensor &network_dst_lookup_ids, const core23::Tensor &network_ev_sizes,
               const core23::Tensor &network_ev_offsets, core23::Tensor &network_comm_buffer,
               const core23::Tensor &d_ev_size_offset, int batch_size, int max_ev_size);

  void dense_backward(const EmbeddingInput &embedding_input, const EmbeddingOutput &top_grad,
                      const DenseNetworkIndices &network_indices,
                      DenseNetworkBuffer &network_buffer, int batch_size);
};

}  // namespace embedding
