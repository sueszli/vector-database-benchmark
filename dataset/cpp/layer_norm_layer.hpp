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

#include <cudnn.h>

#include <general_buffer2.hpp>
#include <memory>
#include <trainable_layer.hpp>

namespace HugeCTR {

/**
 * LayerNorm layer
 */
template <typename T>
class LayerNormLayer : public TrainableLayer<T> {
  using Base = TrainableLayer<T>;

 public:
  /**
   * LayerNorm parameters
   */
  struct Params {
    double eps; /**< small value to avoid divide-by-zero error*/
  };
  /**
   * Ctor of LayerNormLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param params LayerNorm parameters
   * @param cudnn_handle cuDNN handle created externally
   * @param device_id the id of GPU where this layer belongs
   */
  LayerNormLayer(const core23::Tensor& in_tensor, const core23::Tensor& out_tensor,
                 const Params& params, const std::shared_ptr<GPUResource>& gpu_resource,
                 std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());

  /**
   * A method of implementing the forward pass of LayerNorm
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;

  /**
   * A method of implementing the forward pass of LayerNorm
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  /**
   * A method of defining how gamma and beta are initialized.
   * Gamma is initialized to 1s while Beta is 0ed.
   * Override this function to change the initialization behavior.
   */
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;
  const Params params_;

  // these four pointers are just for convenience
  // they are deleted by Layer d'tor through the other pointer aliases: weight_ and wgrad_
  core23::Tensor gamma_;
  core23::Tensor beta_;
  core23::Tensor gamma_grad_;
  core23::Tensor beta_grad_;

  // these tensors are internal only managed by smart ptrs
  core23::Tensor result_save_mean_;
  core23::Tensor result_save_var_;
};

}  // namespace HugeCTR
