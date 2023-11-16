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

#include <core23/tensor.hpp>
#include <data_generator.hpp>
#include <data_readers/data_reader.hpp>
#include <data_readers/data_reader_worker.hpp>
#include <data_readers/file_list.hpp>
#include <fstream>
#include <thread>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

// configuration
const int num_files = 12;
const int batchsize = 65536;
constexpr int iters = 200;
const long long label_dim = 1;
const long long dense_dim = 13;
const long long slot_num = 26;
const long long num_samples_per_file = batchsize * 2 + 10;
const long long num_samples = num_samples_per_file * num_files;
const int max_nnz = 1;
typedef long long T;
const int vocabulary_size = 511;
// configurations for data_reader_worker
const std::string file_list_name("data_reader_file_list.txt");
const std::string prefix("./data_reader_test_data/temp_dataset_");
const Check_t CHK = Check_t::Sum;

void data_reader_worker_norm_test_impl(bool repeat) {
  std::vector<T> generated_sparse_value;
  std::vector<T> generated_sparse_rowoffset;
  std::vector<float> generated_label_data;
  std::vector<float> generated_dense_data;

  if (HugeCTR::file_exist(file_list_name)) {
    remove(file_list_name.c_str());
  }
  // data generation
  HugeCTR::data_generation_for_test<T, CHK>(
      file_list_name, prefix, num_files, num_samples_per_file, slot_num, vocabulary_size, label_dim,
      dense_dim, max_nnz, false, 0.0, &generated_sparse_value, &generated_sparse_rowoffset,
      &generated_label_data, &generated_dense_data);

  ASSERT_TRUE(generated_sparse_rowoffset.size() == num_samples * slot_num);
  ASSERT_TRUE(generated_dense_data.size() == num_samples * dense_dim);
  ASSERT_TRUE(generated_label_data.size() == num_samples * label_dim);

  auto resource_manager = ResourceManagerExt::create({{0}}, 0);
  auto local_gpu = resource_manager->get_local_gpu(0);
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::shared_ptr<ThreadBuffer23> thread_buffer = std::make_shared<ThreadBuffer23>();

  constexpr size_t buffer_length = max_nnz;

  CudaDeviceContext context(0);

  thread_buffer->state.store(BufferState::ReadyForWrite);
  thread_buffer->batch_size = batchsize;
  thread_buffer->param_num = params.size();
  thread_buffer->label_dim = label_dim;
  thread_buffer->dense_dim = dense_dim;
  thread_buffer->batch_size_start_idx = 0;
  thread_buffer->batch_size_end_idx = batchsize;
  for (size_t i = 0; i < params.size(); ++i) {
    auto &param = params[i];
    thread_buffer->is_fixed_length.push_back(params[i].is_fixed_length);
    SparseTensor23 sparse_tensor = SparseTensor23(
        {(int64_t)batchsize, (int64_t)param.max_feature_num}, core23::ToScalarType<T>::value,
        core23::ToScalarType<T>::value, (int64_t)param.slot_num);
    thread_buffer->device_sparse_buffers.push_back(sparse_tensor);
  }
  core23::Tensor label_dense_tensor = core23::Tensor(
      {(int64_t)batchsize, (int64_t)(label_dim + dense_dim)}, core23::ScalarType::Float);
  thread_buffer->device_dense_buffers = label_dense_tensor;
  // setup a data reader
  std::shared_ptr<std::atomic<bool>> loop_flag = std::make_shared<std::atomic<bool>>(1);
  DataReaderWorker<T> data_reader(0, 1, local_gpu, loop_flag, thread_buffer, file_list_name,
                                  buffer_length, repeat, CHK, params);
  // call read a batch
  size_t value_offset = 0;
  int round = (num_samples - 1) / batchsize + 1;

  for (int iter = 0; iter < 50; ++iter) {
    // call read a batch
    data_reader.read_a_batch();
    long long current_batch_size = thread_buffer->current_batch_size;
    if (repeat) {
      ASSERT_TRUE(current_batch_size == batchsize);
    } else {
      if (iter % round == round - 1) {
        ASSERT_TRUE(current_batch_size == num_samples % batchsize);
      } else {
        ASSERT_TRUE(current_batch_size == batchsize);
      }
    }
    auto sparse_tensor = thread_buffer->device_sparse_buffers[0];
    size_t nnz = sparse_tensor.nnz();

    std::unique_ptr<T[]> keys(new T[nnz]);
    HCTR_LIB_THROW(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(), nnz * sizeof(T),
                              cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nnz; ++i) {
      ASSERT_TRUE(generated_sparse_value[(value_offset + i) % generated_sparse_value.size()] ==
                  keys[i])
          << "idx:" << i;
    }

    value_offset += nnz;
    value_offset = value_offset % generated_sparse_value.size();

    std::unique_ptr<T[]> rowoffsets(new T[1 + current_batch_size * slot_num]);
    HCTR_LIB_THROW(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                              (1 + current_batch_size * slot_num) * sizeof(T),
                              cudaMemcpyDeviceToHost));

    T generated_nnz = 0;
    for (int i = 0; i < current_batch_size * slot_num; ++i) {
      T expected = generated_sparse_rowoffset[(iter * batchsize * slot_num + i) %
                                              generated_sparse_rowoffset.size()];

      T value = rowoffsets[i + 1] - rowoffsets[i];
      ASSERT_TRUE(value == expected)
          << "idx:" << i << "value:" << value << " expected:" << expected;
      generated_nnz += expected;
    }

    ASSERT_TRUE(nnz == static_cast<size_t>(generated_nnz));

    auto label_dense_tensor = thread_buffer->device_dense_buffers;

    std::unique_ptr<float[]> label_dense_vec(
        new float[current_batch_size * (dense_dim + label_dim)]);
    HCTR_LIB_THROW(cudaMemcpy(label_dense_vec.get(), label_dense_tensor.data(),
                              current_batch_size * (dense_dim + label_dim) * sizeof(float),
                              cudaMemcpyDeviceToHost));

    for (int i = 0; i < current_batch_size; ++i) {
      int batch_idx = (iter * batchsize + i) % num_samples;
      for (int j = 0; j < label_dim; ++j) {
        ASSERT_FLOAT_EQ(label_dense_vec[i * (label_dim + dense_dim) + j],
                        generated_label_data[batch_idx * label_dim + j]);
      }
      for (int j = 0; j < dense_dim; ++j) {
        float generated_dense = generated_dense_data[batch_idx * dense_dim + j];
        ASSERT_FLOAT_EQ(label_dense_vec[i * (label_dim + dense_dim) + label_dim + j],
                        generated_dense);
      }
    }

    if (!repeat && iter % round == round - 1) {  // need break
      break;
    }
    thread_buffer->state.store(BufferState::ReadyForWrite);
  }
}
void data_reader_norm_test_impl(const std::vector<int> &device_list, int num_threads, bool repeat,
                                bool use_mixed_precision) {
  using DataReaderType = DataReader<T>;
  std::vector<T> generated_sparse_value;
  std::vector<T> generated_sparse_rowoffset;
  std::vector<float> generated_label_data;
  std::vector<float> generated_dense_data;

  if (HugeCTR::file_exist(file_list_name)) {
    remove(file_list_name.c_str());
  }
  // data generation
  HugeCTR::data_generation_for_test<T, CHK>(
      file_list_name, prefix, num_files, num_samples_per_file, slot_num, vocabulary_size, label_dim,
      dense_dim, max_nnz, false, 0.0, &generated_sparse_value, &generated_sparse_rowoffset,
      &generated_label_data, &generated_dense_data);

  std::vector<std::vector<int>> vvgpu{device_list};

  const auto &resource_manager = ResourceManagerExt::create(vvgpu, 0);
  // size_t local_gpu_count = resource_manager->get_local_gpu_count();

  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  DataReaderType data_reader(batchsize, label_dim, dense_dim, params, resource_manager, repeat,
                             num_threads, use_mixed_precision);

  // auto &sparse_tensorbag = data_reader.get_sparse_tensors("distributed");
  data_reader.create_drwg_norm(file_list_name, CHK);
  // int round = (num_samples - 1) / batchsize + 1;
  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iters; ++iter) {
    long long current_batch_size = data_reader.read_a_batch_to_device();
    if (current_batch_size == 0) break;
    HCTR_LOG_S(DEBUG, WORLD) << "iter:" << iter << ", current_batch_size:" << current_batch_size
                             << std::endl;
    if (repeat) {
      ASSERT_TRUE(current_batch_size == batchsize);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  double gbs = ((double)iters * batchsize) / (double)(time_ms / 1000.f);
  std::cout << "Time (ms): " << time_ms << ",  Throughput (samples/s): " << gbs << std::endl;
}

TEST(data_reader_worker, data_reader_worker_test_1) { data_reader_worker_norm_test_impl(true); }

TEST(data_reader_worker, data_reader_worker_test_2) { data_reader_worker_norm_test_impl(false); }

TEST(data_reader_test, data_reader_test_repeat_1) {
  data_reader_norm_test_impl({0}, 1, true, false);
}

TEST(data_reader_test, data_reader_test_repeat_2) {
  data_reader_norm_test_impl({0}, 4, true, false);
}

TEST(data_reader_test, data_reader_test_repeat_3) {
  data_reader_norm_test_impl({0, 1}, 4, true, false);
}

TEST(data_reader_test, data_reader_test_epoch_1) {
  data_reader_norm_test_impl({0}, 1, false, false);
}

TEST(data_reader_test, data_reader_test_epoch_2) {
  data_reader_norm_test_impl({0}, 4, false, false);
}

TEST(data_reader_test, data_reader_test_epoch_3) {
  data_reader_norm_test_impl({0, 1}, 4, false, false);
}

TEST(data_reader_test, legacy_reader_benchmark) {
  data_reader_norm_test_impl({0}, 12, true, false);
}
