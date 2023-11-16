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

#include <general_buffer2.hpp>
#include <optimizer.hpp>

namespace HugeCTR {

/**
 * AdaGrad optimizer
 */
template <typename T>
class AdaGradOptimizer : public Optimizer {
 public:
  /**
   * Constructor of AdaGradOptimizer.
   * names of hyper-parameters are the same as in AdaGrad paper
   * (https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
   * @param weight_tensors a list of weights in dense layers
   * @param wgrad_tensors a list of wgrad tensors
   * @param gpu_resource the GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param initial_accu_value  initial value for the accumulation
   * @param epsilon
   * @param scaler scaler for gradient values
   */
  AdaGradOptimizer(std::optional<WeightTensors> weight_tensors,
                   std::optional<WgradTensors<T>> wgrad_tensors,
                   const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                   float initial_accu_value = 0., float epsilon = 1e-7, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

  std::vector<core23::Tensor> get_opt_state_tensors() override { return {accum_tensor_}; }

 private:
  std::optional<WgradTensors<T>> wgrad_tensors_;
  core23::Tensor accum_tensor_;

  float initial_accumulator_value_;
  const float epsilon_;
};

}  // namespace HugeCTR
