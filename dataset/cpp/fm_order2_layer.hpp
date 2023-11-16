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

namespace HugeCTR {

/**
 * The order2 expression in FM formular(reference paper:
 * https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf).
 * The layer will be used in DeepFM model to implement the FM order2
 * computation (reference code implemented in Tensorflow: line 92~104,
 * https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py).
 */
template <typename T>
class FmOrder2Layer : public Layer {
 public:
  FmOrder2Layer(const core23::Tensor& input_tensor, const core23::Tensor& output_tensor,
                const std::shared_ptr<GPUResource>& gpu_resource);
  /**
   * A method of implementing the forward pass of FmOrder2
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train);

  /**
   * A method of implementing the backward pass of FmOrder2
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop();

 private:
  int batch_size_;
  int slot_num_;
  int embedding_vec_size_;
};

}  // namespace HugeCTR
