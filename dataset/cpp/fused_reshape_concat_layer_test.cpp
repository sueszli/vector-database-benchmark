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

#include <layers/fused_reshape_concat_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <typename T>
void fused_reshape_concat_cpu(bool forward, T *output_item, T *output_ad, T **inputs,
                              size_t batch_size, size_t slot_num, int *vecs_size, int num,
                              size_t out_vector_size) {
  for (size_t l = 0; l < batch_size; l++) {
    for (size_t i = 0; i < slot_num; i++) {
      int count = 0;
      for (int j = 0; j < num; j++) {
        for (int k = 0; k < vecs_size[j]; k++) {
          if (forward) {
            if (i == slot_num - 1)
              output_ad[l * out_vector_size + count] =
                  inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k];
            else
              output_item[l * (slot_num - 1) * out_vector_size + i * out_vector_size + count] =
                  inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k];
          } else {
            if (i == slot_num - 1)
              inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k] =
                  output_ad[l * out_vector_size + count];
            else
              inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k] =
                  output_item[l * (slot_num - 1) * out_vector_size + i * out_vector_size + count];
          }
          count++;
        }
      }
    }
  }
}
template <typename T>
void core23_fused_reshape_concat_test(int64_t batch_size, int64_t slot_num,
                                      std::vector<int> items) {
  int num = items.size();
  int64_t out_vector_size = 0;
  int *vecs_size = new int[num];
  std::vector<core23::Tensor> in_tensors;

  for (int i = 0; i < num; i++) {
    int64_t embedding_vec_size = items[i];
    core23::Shape dims_in{batch_size, slot_num, embedding_vec_size};
    core23::Tensor in_tensor(core23::TensorParams()
                                 .shape(dims_in)
                                 .data_type(core23::ToScalarType<T>::value)
                                 .device({core23::DeviceType::GPU, 0}));
    in_tensors.push_back(in_tensor);
    out_vector_size += embedding_vec_size;
    vecs_size[i] = embedding_vec_size;
  }

  std::vector<core23::Tensor> out_tensors;
  int64_t rows = batch_size * slot_num;
  int64_t out_size_item = batch_size * (slot_num - 1) * out_vector_size;
  int64_t out_size_ad = batch_size * out_vector_size;
  FusedReshapeConcatLayer<T> fused_reshape_concat_layer(in_tensors, out_tensors,
                                                        test::get_default_gpu());

  fused_reshape_concat_layer.initialize();
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].data<T>();
  }

  T *d_out_item = out_tensors[0].data<T>();
  T *d_out_ad = out_tensors[1].data<T>();
  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins[i] = new T[rows * items[i]];
  }

  std::unique_ptr<T *[]> h_ins_b(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins_b[i] = new T[rows * items[i]];
  }
  std::unique_ptr<T[]> h_out_item(new T[out_size_item]);
  std::unique_ptr<T[]> h_out_ad(new T[out_size_ad]);
  std::unique_ptr<T[]> h_cpu_out_item(new T[out_size_item]);
  std::unique_ptr<T[]> h_cpu_out_ad(new T[out_size_ad]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fused_reshape_concat_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(
      cudaMemcpy(h_out_item.get(), d_out_item, out_size_item * sizeof(T), cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(
      cudaMemcpy(h_out_ad.get(), d_out_ad, out_size_ad * sizeof(T), cudaMemcpyDeviceToHost));

  fused_reshape_concat_cpu(true, h_cpu_out_item.get(), h_cpu_out_ad.get(), h_ins.get(), batch_size,
                           slot_num, vecs_size, num, out_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out_ad.get(), h_cpu_out_ad.get(), out_size_ad,
                                            Eps<T>::value()));
  ASSERT_TRUE(test::compare_array_approx<T>(h_out_item.get(), h_cpu_out_item.get(), out_size_item,
                                            Eps<T>::value()));

  // bprop
  test::GaussianDataSimulator simulatorb(0.0f, 2.0f);
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    memset(h_ins[i], 0, size * sizeof(T));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulatorb.fill(h_out_item.get(), out_size_item);
  simulatorb.fill(h_out_ad.get(), out_size_ad);
  HCTR_LIB_THROW(
      cudaMemcpy(d_out_item, h_out_item.get(), out_size_item * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_out_ad, h_out_ad.get(), out_size_ad * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fused_reshape_concat_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    HCTR_LIB_THROW(cudaMemcpy(h_ins_b[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  fused_reshape_concat_cpu(false, h_out_item.get(), h_out_ad.get(), h_ins.get(), batch_size,
                           slot_num, vecs_size, num, out_vector_size);
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    ASSERT_TRUE(test::compare_array_approx<T>(h_ins[i], h_ins_b[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

}  // namespace

TEST(core23_fused_reshape_concat_layer, fp32_32x20x12_3) {
  std::vector<int> items;
  int batch_size = 32, slot_num = 20;
  int goodID_size = 3, shopID_size = 5, cateID_size = 4;
  items.push_back(goodID_size);
  items.push_back(shopID_size);
  items.push_back(cateID_size);
  core23_fused_reshape_concat_test<float>(batch_size, slot_num, items);
}

TEST(core23_fused_reshape_concat_layer, fp32_32x20_7) {
  std::vector<int> items{21, 4, 7, 13, 75, 34, 13};
  core23_fused_reshape_concat_test<float>(32, 20, items);
}

TEST(core23_fused_reshape_concat_layer, fp32_32x100_16) {
  std::vector<int> items{21, 4, 7, 13, 75, 34, 13, 23, 76, 34, 13, 12, 14, 5, 8, 20};
  core23_fused_reshape_concat_test<float>(32, 100, items);
}

TEST(core23_fused_reshape_concat_layer, fp32_128x200_16) {
  std::vector<int> items{21, 54, 27, 13, 75, 34, 13, 23, 76, 34, 13, 12, 14, 5, 8, 20};
  core23_fused_reshape_concat_test<float>(128, 200, items);
}

TEST(core23_fused_reshape_concat_layer, fp32_128x1024_11) {
  std::vector<int> items{211, 54, 270, 130, 75, 34, 131, 231, 76, 341, 130};
  core23_fused_reshape_concat_test<float>(128, 1024, items);
}