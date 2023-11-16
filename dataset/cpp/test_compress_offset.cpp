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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <embedding/operators/compress_offset.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <utils.hpp>

using namespace embedding;

TEST(test_compress_offset, test_compress_offset) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  HugeCTR::core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  HugeCTR::core23::TensorParams params = core23::TensorParams().device(device);

  int batch_size = 5;
  int num_table = 2;
  int num_offset = batch_size * num_table + 1;
  int num_compressed_offset = num_table + 1;

  auto offset = core23::Tensor(params.shape({num_offset}).data_type(core23::ScalarType::UInt32));

  std::vector<uint32_t> cpu_offset{0};
  for (int i = 1; i < num_offset; ++i) {
    int n = rand() % 10;
    cpu_offset.push_back(n);
  }
  std::partial_sum(cpu_offset.begin(), cpu_offset.end(), cpu_offset.begin());
  core23::copy_sync(offset, cpu_offset);

  CompressOffset compress_offset{core, num_compressed_offset, offset.data_type()};
  HugeCTR::core23::Tensor compressed_offset;
  compress_offset.compute(offset, batch_size, &compressed_offset);

  HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));

  std::vector<uint32_t> gpu_compressed_offset(compressed_offset.num_elements());
  HugeCTR::core23::copy_sync(gpu_compressed_offset, compressed_offset);

  ASSERT_EQ(gpu_compressed_offset.size(), num_compressed_offset);
  for (int i = 0; i < num_compressed_offset; ++i) {
    ASSERT_EQ(gpu_compressed_offset[i], cpu_offset[i * batch_size]);
  }
}