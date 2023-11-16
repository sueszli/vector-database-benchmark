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
#include <omp.h>

#include <common.hpp>
#include <cstdio>
#include <data_readers/async_reader/async_reader.hpp>
#include <fstream>
#include <functional>
#include <general_buffer2.hpp>
#include <iostream>
#include <resource_managers/resource_manager_ext.hpp>
#include <sstream>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

void reader_test(std::vector<int> device_list, size_t file_size, size_t batch_size, int num_threads,
                 int batches_per_thread, int io_block_size, int io_depth, int wait_time_us) {
  const std::string fname = "__tmp_test.dat";
  char* ref_data;
  char* read_data;

  HCTR_LIB_THROW(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 424242);

  HCTR_LIB_THROW(cudaMallocManaged(&ref_data, file_size));
  HCTR_LIB_THROW(cudaMallocManaged(&read_data, file_size));

#pragma omp parallel
  {
    std::mt19937 gen(424242 + omp_get_thread_num());
    // std::uniform_int_distribution<uint8_t> dis(0, 255);
    std::uniform_int_distribution<uint8_t> dis('a', 'z');

#pragma omp for
    for (size_t i = 0; i < file_size; i++) {
      ref_data[i] = dis(gen);
    }
  }

  {
    std::ofstream fout(fname);
    fout.write(ref_data, file_size);
  }

  AsyncReaderImpl reader_impl(fname, batch_size, resource_manager.get(), num_threads,
                              batches_per_thread, io_block_size, io_depth, 4096);

  reader_impl.load_async();

  size_t total_sz = 0;
  while (true) {
    BatchDesc desc = reader_impl.get_batch();
    size_t sz = desc.size_bytes;

    if (sz > 0) {
      HCTR_LIB_THROW(
          cudaMemcpy(read_data + total_sz, desc.dev_data[0], sz, cudaMemcpyDeviceToDevice));
      total_sz += sz;
      usleep(wait_time_us);
      reader_impl.finalize_batch();
    } else {
      break;
    }
    if (total_sz >= file_size) {
      break;
    }
  }

  ASSERT_EQ(total_sz, file_size);
  for (size_t i = 0; i < std::min(file_size, total_sz); i++) {
    // HCTR_LOG_S(DEBUG, WORLD) << "Symbols differ at index " << i << " : expected "
    //           << ref_data[i] << " got " << read_data[i] << std::endl;
    ASSERT_EQ(ref_data[i], read_data[i]) << "Symbols differ at index " << i << " : expected "
                                         << ref_data[i] << " got " << read_data[i];
  }

  cudaFree(ref_data);
  cudaFree(read_data);
}

//   device_list   file_size batch  threads  batch_per_thread  io_block  io_depth  wait_time
//
TEST(reader_test, test1) { reader_test({0}, 100, 20, 1, 1, 4096 * 2, 1, 0); }
TEST(reader_test, test2) { reader_test({0}, 100, 20, 2, 1, 4096 * 2, 1, 0); }
TEST(reader_test, test3) { reader_test({0}, 1012, 20, 2, 1, 4096 * 2, 1, 0); }
TEST(reader_test, test4) { reader_test({0}, 1012, 32, 2, 2, 4096 * 2, 1, 0); }
TEST(reader_test, test5) { reader_test({0}, 10120, 32, 2, 2, 4096 * 2, 2, 0); }
TEST(reader_test, test6) { reader_test({0}, 101256, 1000, 2, 4, 4096 * 2, 2, 0); }
TEST(reader_test, test7) { reader_test({0}, 101256, 1000, 2, 4, 4096 * 2, 2, 100); }
TEST(reader_test, test8) { reader_test({0}, 101256, 1000, 2, 4, 4096 * 2, 2, 1000); }
TEST(reader_test, test9) { reader_test({0, 1}, 100, 20, 2, 1, 4096 * 2, 1, 0); }
TEST(reader_test, test10) { reader_test({0, 1}, 101256, 1000, 2, 4, 4096 * 2, 2, 0); }
TEST(reader_test, test11) { reader_test({0, 1}, 101256, 1000, 2, 4, 4096 * 2, 2, 100); }
TEST(reader_test, test12) { reader_test({0, 1}, 101256, 1000, 2, 4, 4096 * 2, 2, 1000); }
TEST(reader_test, test13) { reader_test({0, 1}, 1014252, 14352, 6, 4, 4096 * 2, 2, 0); }
TEST(reader_test, test14) { reader_test({0, 1, 2, 3}, 100980, 1980, 4, 4, 4096 * 2, 2, 1000); }
TEST(reader_test, test15) { reader_test({0, 1, 2, 3, 4}, 101256, 7616, 8, 4, 4096 * 2, 2, 0); }
TEST(reader_test, test16) {
  reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 8012516, 38720, 8, 4, 4096 * 2, 2, 0);
}
TEST(reader_test, test17) {
  reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 8012516, 38720, 16, 4, 4096 * 2, 2, 0);
}
TEST(reader_test, test18) {
  reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 18012516, 38720, 8, 4, 4096 * 2, 2, 2000);
}
