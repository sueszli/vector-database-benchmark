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

#include <cublasLt.h>
#include <cublas_v2.h>

#include <functional>
#include <layer.hpp>
#include <layers/functors/fused_fc_layer_functors.hpp>
#include <trainable_layer.hpp>
#include <vector>

namespace HugeCTR {

template <typename T>
class MLPLayer : public TrainableLayer<T> {
  std::vector<core23::Tensor> train_tensors_, mask_tensors_, dact_tensors_, db_tensors_;

  std::vector<core23::Tensor> kernels_;
  std::vector<core23::Tensor> biases_;
  std::vector<core23::Tensor> kernels_grad_;

  std::vector<int64_t> num_outputs_;
  std::vector<Activation_t> acts_;

  std::vector<bool> output_mask_;
  std::vector<bool> use_bias_;

  bool async_wgrad_;
  bool fuse_wb_;
  bool enable_tf32_compute_;
  bool skip_head_dgrad_;

  bool event_overlap_created_;
  cudaEvent_t event_overlap_;
  std::vector<CublasFusedFCLayerDesc<T>> layer_desc_;
  std::vector<CublasFusedFCLayerAlgo<T>> layer_algo_;
  FusedFCLayerFunctors<T> layer_functors_;

  std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;

 public:
  MLPLayer(const std::vector<core23::Tensor>& bottom_tensors,
           const std::vector<core23::Tensor>& top_tensors, const std::vector<int64_t>& num_outputs,
           const std::shared_ptr<GPUResource>& gpu_resource, const std::vector<Activation_t>& acts,
           const std::vector<bool>& use_bias,
           std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>(),
           bool skip_head_dgrad = false, bool async_wgrad = false, bool fuse_wb = false,
           bool enable_tf32_compute = false);

  MLPLayer(const MLPLayer& C) = delete;
  MLPLayer& operator=(const MLPLayer&);

  void fprop(bool is_train) final;

  void bprop() final;

  void search_algorithm() final;

  void initialize() final;

  /*
   * Interfaces for unit tests to debug
   */
  auto& get_kernel(int index) { return kernels_[index]; }
  auto& get_bias(int index) { return biases_[index]; }
  auto& get_kernel_grad(int index) { return kernels_grad_[index]; }
  auto& get_bias_grad(int index) { return db_tensors_[index]; }
  auto& get_inner_tensors() { return train_tensors_; }
  auto& get_input_tensors() { return this->input_tensors_; }
  auto& get_output_tensors() { return this->output_tensors_; }

  ~MLPLayer() {
    CudaDeviceContext context(this->get_device_id());
    if (event_overlap_created_) {
      cudaEventDestroy(event_overlap_);
    }
  };
};
}  // namespace HugeCTR
