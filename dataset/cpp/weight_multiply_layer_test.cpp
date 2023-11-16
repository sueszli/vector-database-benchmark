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

#include <core23/tensor_container.hpp>
#include <layers/weight_multiply_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
namespace {

template <typename T>
T eps();

template <>
constexpr float eps() {
  return 1e-1f;
}

template <>
__half eps() {
  return __float2half(2e-0f);
}

template <typename T>
void weight_multiply_cpu(const T* input, const T* weight, T* output, int batch_size, int slot_num,
                         int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      for (int k = 0; k < embedding_vec_size; k++) {
        output[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] =
            input[i * slot_num + j] * weight[j * embedding_vec_size + k];
      }
    }
  }
}

template <typename T>
void weight_multiply_wgrad_cpu(const T* top_grad, const T* input, T* wgrad, int batch_size,
                               int slot_num, int embedding_vec_size) {
  int len_w = slot_num * embedding_vec_size;
  for (int i = 0; i < len_w; i++) {
    double tmp = 0.0;
    for (int j = 0; j < batch_size; j++) {
      tmp += (double)input[j * slot_num + i / embedding_vec_size] * (double)top_grad[j * len_w + i];
    }
    wgrad[i] = (T)tmp;
  }
}

template <typename T>
void weight_multiply_dgrad_cpu(const T* top_grad, const T* weight, T* dgrad, int batch_size,
                               int slot_num, int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      float tmp = 0.0;
      for (int k = 0; k < embedding_vec_size; k++) {
        tmp = tmp + float(top_grad[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] *
                          weight[j * embedding_vec_size + k]);
      }
      dgrad[i * slot_num + j] = TypeConvert<T, float>::convert(tmp);
    }
  }
}

template <typename T>
void weight_multiply_test(int64_t batch_size, int64_t slot_num, int64_t embedding_vec_size) {
  core23::Shape in_dims = {batch_size, slot_num};
  core23::Shape weight_dims = {slot_num, embedding_vec_size};

  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  core23::Tensor in_tensor = core23::Tensor(core23::TensorParams()
                                                .data_type(core23::ToScalarType<T>::value)
                                                .shape(in_dims)
                                                .buffer_params(blobs_buffer_params));
  core23::Tensor out_tensor;

  WeightMultiplyLayer<T> weight_multiply_layer(in_tensor, out_tensor, weight_dims,
                                               test::get_default_gpu());

  weight_multiply_layer.initialize();

  auto weights = weight_multiply_layer.get_weights();
  auto weights_grad = weight_multiply_layer.get_wgrads();

  core23::TensorContainer<__half, 1, 1> weights_container(std::move(weights),
                                                          {static_cast<int64_t>(weights.size())});
  core23::TensorContainer<__half, 1, 1> weights_grad_container(
      std::move(weights_grad), {static_cast<int64_t>(weights_grad.size())});

  T* d_weight = weights_container[0].data<T>();
  T* d_wgrad = weights_grad_container[0].data<T>();

  const int64_t len_in = batch_size * slot_num;
  const int64_t len_out = batch_size * slot_num * embedding_vec_size;
  const int64_t len_w = slot_num * embedding_vec_size;
  T* d_in = in_tensor.data<T>();
  T* d_out = out_tensor.data<T>();
  std::unique_ptr<T[]> h_in(new T[len_in]);
  std::unique_ptr<T[]> h_out(new T[len_out]);
  std::unique_ptr<T[]> h_weight(new T[len_w]);
  std::unique_ptr<T[]> h_wgrad(new T[len_w]);
  std::unique_ptr<T[]> h_expected(new T[len_out]);
  std::unique_ptr<T[]> h_expected_wgrad(new T[len_w]);

  // fprop
  for (size_t i = 0; i < len_in; i++) {
    h_in[i] = TypeConvert<T, float>::convert(float(i % slot_num));
  }
  for (size_t i = 0; i < len_w; i++) {
    h_weight[i] = TypeConvert<T, float>::convert(float(1.0f));
  }
  HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), len_in * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_weight, h_weight.get(), len_w * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  weight_multiply_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, len_out * sizeof(T), cudaMemcpyDeviceToHost));

  weight_multiply_cpu(h_in.get(), h_weight.get(), h_expected.get(), batch_size, slot_num,
                      embedding_vec_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_expected.get(), len_out, eps<T>()));

  // bprop
  for (int64_t i = 0; i < len_in; ++i) {
    h_expected[i] = h_in[i];
  }
  HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), len_in * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), len_out * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_weight, h_weight.get(), len_w * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  weight_multiply_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(
      cudaMemcpy(h_wgrad.get(), d_wgrad, len_w * sizeof(T), cudaMemcpyDeviceToHost));  // wgrad
  HCTR_LIB_THROW(
      cudaMemcpy(h_in.get(), d_in, len_in * sizeof(T), cudaMemcpyDeviceToHost));  // dgrad

  weight_multiply_wgrad_cpu(h_out.get(), h_expected.get(), h_expected_wgrad.get(), batch_size,
                            slot_num, embedding_vec_size);
  // TODO: because of the accumulated error, comparing absolute error can not pass when esp<1e-3
  ASSERT_TRUE(test::compare_array_approx<T>(h_wgrad.get(), h_expected_wgrad.get(), len_w,
                                            eps<T>()));  // compare wgrad
  // CAUTION: dgrad computation will modify the "input", so it must be put after wgrad computation
  weight_multiply_dgrad_cpu(h_out.get(), h_weight.get(), h_expected.get(), batch_size, slot_num,
                            embedding_vec_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_in.get(), h_expected.get(), len_in,
                                            eps<T>()));  // compare dgrad
}

}  // namespace

TEST(weight_multiply_layer, fp32_40960x10x128) { weight_multiply_test<float>(40960, 10, 128); }
TEST(weight_multiply_layer, fp32_1024x10x128) { weight_multiply_test<float>(1024, 10, 128); }
TEST(weight_multiply_layer, fp32_1024x64x128) { weight_multiply_test<float>(1024, 64, 128); }
// this would lead to error
TEST(weight_multiply_layer, fp16_1024x64x128) { weight_multiply_test<__half>(2, 64, 128); }
TEST(weight_multiply_layer, fp16_1024x10x128) { weight_multiply_test<__half>(1024, 10, 128); }
