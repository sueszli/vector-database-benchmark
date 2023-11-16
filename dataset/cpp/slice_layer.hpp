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

#include <layer.hpp>
#include <set>
#include <vector>

namespace HugeCTR {

/**
 * Layer which splits a single 2D input tensor into multiple 2D output tensors across columns.
 * e.g., (batch_size, 90) to (batch_size, 40) and (batch_size, 4) by choosing the column ranges
 * [0:40) and (50:90). It is possible those ranges overlap, e.g., [0:100) and [50:200).
 */
template <typename T>
class SliceLayer : public Layer {
 public:
  SliceLayer(const core23::Tensor& input_tensor, std::vector<core23::Tensor>& output_tensors,
             std::vector<std::pair<int, int>>& ranges,
             const std::shared_ptr<GPUResource>& gpu_resource);
  ~SliceLayer() override{};

  /**
   * Slice's forward pass to gather data to the output tensor
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * Slice's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  std::vector<int> slices_start_;
};

}  // namespace HugeCTR
