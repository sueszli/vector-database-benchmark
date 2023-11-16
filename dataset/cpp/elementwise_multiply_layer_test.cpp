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

#include <gtest/gtest.h>

#include <layers/elementwise_multiply_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
T eps();

template <>
constexpr float eps() {
  return 1e-5f;
}

template <>
__half eps() {
  return __float2half(1e-3f);
}

template <typename T>
void elementwise_multiply_cpu(T **input, T *output, size_t size, size_t num) {
  T one = 1.0;

  for (size_t i = 0; i < size; i++) {
    T tmp = one;
    for (size_t j = 0; j < num; j++) {
      tmp = tmp * input[j][i];
    }
    output[i] = tmp;
  }
}

template <typename T>
void elementwise_multiply_dgrad_cpu(const T *top_grad, T **dgrad, const T *fprop_output,
                                    size_t size, size_t num) {
  T zero = 0.0;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      if (0 == __half2float(fprop_output[i])) {
        dgrad[j][i] = zero;
      } else {
        T d_input = dgrad[j][i];
        dgrad[j][i] = top_grad[i] * T(fprop_output[i] / d_input);
      }
    }
  }
}

template <typename T>
void core23_elementwise_multiply_test(int64_t batch_size, int64_t slot_num,
                                      int64_t embedding_vec_size, int64_t num) {
  core23::Shape dims_in{batch_size, slot_num, embedding_vec_size};
  core23::Shape dims_out{batch_size, slot_num, embedding_vec_size};
  int64_t size = batch_size * slot_num * embedding_vec_size;

  std::vector<core23::Tensor> in_tensors;
  for (size_t i = 0; i < num; i++) {
    core23::Tensor tensor(core23::TensorParams()
                              .shape(dims_in)
                              .data_type(core23::ToScalarType<T>::value)
                              .device({core23::DeviceType::GPU, 0}));
    in_tensors.push_back(tensor);
  }
  core23::Tensor out_tensor(core23::TensorParams()
                                .shape(dims_out)
                                .data_type(core23::ToScalarType<T>::value)
                                .device({core23::DeviceType::GPU, 0}));

  ElementwiseMultiplyLayer<T> elementwise_multiply_layer(in_tensors, out_tensor,
                                                         test::get_default_gpu());

  elementwise_multiply_layer.initialize();

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].data<T>();
  }
  T **d_ins;
  HCTR_LIB_THROW(cudaMalloc((void **)(&d_ins), num * sizeof(T *)));
  HCTR_LIB_THROW(
      cudaMemcpy((void *)d_ins, (void *)h_d_ins.get(), num * sizeof(T *), cudaMemcpyHostToDevice));
  T *d_out = out_tensor.data<T>();

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new T[size];
  }
  std::unique_ptr<T[]> h_out(new T[size]);
  std::unique_ptr<T[]> fprop_output(new T[size]);
  std::unique_ptr<T[]> h_cpu_out(new T[size]);
  std::unique_ptr<T *[]> h_gpu_dgrads(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_gpu_dgrads[i] = new T[size];
  }

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  elementwise_multiply_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(fprop_output.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));

  elementwise_multiply_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), size, eps<T>()));

  // bprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulator.fill(h_out.get(), size);
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  elementwise_multiply_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (size_t i = 0; i < num; i++) {
    HCTR_LIB_THROW(
        cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  elementwise_multiply_dgrad_cpu(h_out.get(), h_ins.get(), fprop_output.get(), size, num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(
        test::compare_array_approx<T>(h_ins[i], h_gpu_dgrads[i], size, eps<T>()));  // compare dgrad
  }
}

}  // namespace

TEST(core23_elementwise_multiply_layer, fp32_40960x1x1) {
  core23_elementwise_multiply_test<float>(40960, 1, 1, 3);
}
TEST(core23_elementwise_multiply_layer, fp16_40960x1x1) {
  core23_elementwise_multiply_test<__half>(40960, 1, 1, 3);
}
TEST(core23_elementwise_multiply_layer, fp32_40960x4x3) {
  core23_elementwise_multiply_test<float>(40960, 4, 3, 3);
}
TEST(core23_elementwise_multiply_layer, fp16_40960x4x3) {
  core23_elementwise_multiply_test<__half>(40960, 4, 3, 3);
}
TEST(core23_elementwise_multiply_layer, fp32_4096x4x256) {
  core23_elementwise_multiply_test<float>(4096, 4, 256, 3);
}
TEST(core23_elementwise_multiply_layer, fp16_4096x4x256) {
  core23_elementwise_multiply_test<__half>(4096, 4, 256, 3);
}
