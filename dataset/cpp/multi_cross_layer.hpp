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

#include <functional>
#include <layers/functors/fused_fc_layer_functors.hpp>
#include <trainable_layer.hpp>
#include <vector>

namespace HugeCTR {

template <typename T>
struct MultiCrossForwardFunctor {
  MultiCrossForwardFunctor() = default;
  MultiCrossForwardFunctor(const MultiCrossForwardFunctor&) = delete;
  MultiCrossForwardFunctor& operator=(const MultiCrossForwardFunctor&) = delete;

  void operator()(cudaStream_t stream, cublasHandle_t cublas_handle,
                  const core23::Tensor& input_tensor,
                  const std::vector<core23::Tensor>& kernel_tensors,
                  const std::vector<core23::Tensor>& bias_tensors,
                  std::vector<core23::Tensor>& layer_output_tensors,
                  std::vector<core23::Tensor>& layer_hidden_tensors, int num_layers) const;
};
template <typename T>
struct MultiCrossForwardFunctorv2 {
  GemmFunctor<T> gemm_functor_;
  MultiCrossForwardFunctorv2() = default;
  MultiCrossForwardFunctorv2(const MultiCrossForwardFunctorv2&) = delete;
  MultiCrossForwardFunctorv2& operator=(const MultiCrossForwardFunctorv2&) = delete;
  void search_algorithm(T* bottom, T* top, T* kernel, int64_t batch_size, int64_t input_size,
                        int64_t output_size, const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                        cublasLtHandle_t cublaslt_handle, cudaStream_t stream);
  void operator()(cudaStream_t stream, const core23::Tensor& input_tensor,
                  const std::vector<core23::Tensor>& kernel_tensors,
                  const std::vector<core23::Tensor>& bias_tensors,
                  std::vector<core23::Tensor>& XU_tensors,
                  std::vector<core23::Tensor>& layer_output_tensors,
                  std::vector<core23::Tensor>& layer_hidden_tensors, int num_layers,
                  const std::vector<CublasDesc<T>>& xu_descr_,
                  const std::vector<CublasDesc<T>>& xuvb_descr_,
                  const std::vector<CublasAlgo<T>>& xu_fprop_algo_,
                  const std::vector<CublasAlgo<T>>& xuvb_fprop_algo_, cublasLtHandle_t = nullptr);
};

template <typename T>
struct MultiCrossBackwardFunctorv2 {
  GemmFunctor<T> gemm_functor_;

  MultiCrossBackwardFunctorv2() = default;
  MultiCrossBackwardFunctorv2(const MultiCrossBackwardFunctorv2&) = delete;
  MultiCrossBackwardFunctorv2& operator=(const MultiCrossBackwardFunctorv2&) = delete;
  void operator()(cudaStream_t dgrad_stream, cudaStream_t wgrad_stream, bool async_wgrad,
                  cudaEvent_t& event_overlap, const core23::Tensor& input_tensor,
                  const std::vector<core23::Tensor>& kernel_tensors,
                  const std::vector<core23::Tensor>& act_tensors,
                  const std::vector<core23::Tensor>& layer_hidden_tensors,
                  std::vector<core23::Tensor>& kernel_output_tensors,
                  std::vector<core23::Tensor>& grad_tensors,
                  std::vector<core23::Tensor>& bias_output_tensors,
                  std::vector<core23::Tensor>& XU_tensors, core23::Tensor accum_dx_tensor_,
                  std::vector<core23::Tensor> bprop_bottoms, int num_layers,
                  const std::vector<CublasDesc<T>>& xu_descr_,
                  const std::vector<CublasDesc<T>>& xuvb_descr_,
                  const std::vector<CublasDesc<T>>& du_descrs_bprop_,
                  const std::vector<CublasDesc<T>>& dhidden_descrs_bprop_,
                  const std::vector<CublasAlgo<T>>& xu_bprop_algo_,
                  const std::vector<CublasAlgo<T>>& xuvb_bprop_algo_,
                  const std::vector<CublasAlgo<T>>& du_bprop_algos_,
                  const std::vector<CublasAlgo<T>>& dhidden_bprop_algos_,
                  cublasLtHandle_t cublaslt_handle = nullptr);
};

template <typename T>
struct MultiCrossBackwardFunctor {
  MultiCrossBackwardFunctor() = default;
  MultiCrossBackwardFunctor(const MultiCrossBackwardFunctor&) = delete;
  MultiCrossBackwardFunctor& operator=(const MultiCrossBackwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const core23::Tensor& input_tensor,
                  const std::vector<core23::Tensor>& kernel_tensors,
                  const std::vector<core23::Tensor>& layer_output_tensors,
                  const std::vector<core23::Tensor>& layer_hidden_tensors,
                  const core23::Tensor& grad_tensor, core23::Tensor& output_tensor,
                  std::vector<core23::Tensor>& kernel_output_tensors,
                  std::vector<core23::Tensor>& bias_output_tensors, core23::Tensor& tmp_vec_tensor,
                  core23::Tensor tmp_mat_tensors[], int num_layers) const;
};

template <typename T>
class MultiCrossLayer : public TrainableLayer<T> {
 private:
  const int num_layers_;
  const int64_t projection_dim_;

  std::vector<core23::Tensor> dgrads_;
  std::vector<core23::Tensor> activation_tensors_; /**< vector of internal blobs' tensors,
                                intermediate output of each    interaction layer: T_4 */
  std::vector<core23::Tensor> hidden_tensors_;     // DCNv1: x_i * w ; DCNv2: x * x_i * w + b; T_7
  std::vector<core23::Tensor> XU_tensors_;         // DCNv2:

  core23::Tensor tmp_mat_tensors_[4];  //[h,w]

  core23::Tensor accum_dx_tensor_;
  std::vector<core23::Tensor> bprop_bottom_;
  core23::Tensor tmp_vec_tensor_;  //[h,1]

  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<core23::Tensor> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<core23::Tensor> out_tensors_;

  std::vector<CublasDesc<T>> xu_descrs_fprop_;
  std::vector<CublasDesc<T>> xuvb_descrs_fprop_;
  std::vector<CublasDesc<T>> xu_descrs_bprop_;
  std::vector<CublasDesc<T>> xuvb_descrs_bprop_;
  std::vector<CublasDesc<T>> du_descrs_bprop_;
  std::vector<CublasDesc<T>> dhidden_descrs_bprop_;

  std::vector<CublasAlgo<T>> xu_fprop_algos_;
  std::vector<CublasAlgo<T>> xuvb_fprop_algos_;
  std::vector<CublasAlgo<T>> xu_bprop_algos_;
  std::vector<CublasAlgo<T>> xuvb_bprop_algos_;
  std::vector<CublasAlgo<T>> du_bprop_algos_;
  std::vector<CublasAlgo<T>> dhidden_bprop_algos_;

  MultiCrossForwardFunctorv2<T> dcnv2_forward_functor_;
  MultiCrossBackwardFunctorv2<T> dcnv2_backward_functor_;
  bool enable_tf32_compute_;
  bool async_wgrad_ = false;
  cudaStream_t wgrad_stream_;
  cudaEvent_t event_fork_;

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train) final;
  std::vector<core23::Tensor>& get_hidden_tensors() { return hidden_tensors_; };
  std::vector<core23::Tensor>& get_weight_tensor() { return XU_tensors_; };
  /**
   * backward pass
   */
  void search_algorithm() override;
  void bprop() final;
  void initialize() override;
  MultiCrossLayer(const std::vector<core23::Tensor>& in_tensors,
                  const std::vector<core23::Tensor>& out_tensors,
                  const std::shared_ptr<GPUResource>& gpu_resource, int num_layers,
                  int64_t projection_dim,
                  std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>(),
                  bool enable_tf32_compute = false, bool async_wgrad = false);
  MultiCrossLayer(const MultiCrossLayer&) = delete;
  MultiCrossLayer& operator=(const MultiCrossLayer&) = delete;

 private:
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;
};

}  // namespace HugeCTR
