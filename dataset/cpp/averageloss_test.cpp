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

#include <HugeCTR/include/network_buffer_channels.hpp>
#include <cstdio>
#include <fstream>
#include <functional>
#include <metrics.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <sstream>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 5.0e-2;

template <typename T>
void gen_random(std::vector<float>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);
  std::normal_distribution<float> dis_neg(0, 0.5);
  std::normal_distribution<float> dis_pos(1, 0.5);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    int label = dis_label(gen);
    h_labels[i] = (float)label;

    h_scores[i] = (T)-1.0;
    while (!((T)0.0 <= h_scores[i] && h_scores[i] <= (T)1.0)) {
      h_scores[i] = (float)(label ? dis_pos(gen) : dis_neg(gen));
    }
  }
}

template <typename T>
void gen_same(std::vector<float>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = 0.2345;
  }
}

template <typename T>
void gen_correct(std::vector<float>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = (T)h_labels[i];
  }
}

template <typename T>
void gen_wrong(std::vector<float>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = (T)(1.0f - h_labels[i]);
  }
}

template <typename T>
void gen_multilobe(std::vector<T>& h_labels, std::vector<T>& h_scores, int offset) {
  const int npeaks = 2;
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);
  std::uniform_int_distribution<int> dis_score(1, npeaks);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = (float)dis_score(gen) / ((float)npeaks + 1);
  }
}

static int execution_number = 0;

template <typename T, typename Generator>
void averageloss_test(std::vector<int> device_list, size_t batch_size, size_t num_total_samples,
                      Generator gen, size_t num_evals = 1) {
  int num_procs = 1, rank = 0;
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
#endif
  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  std::vector<std::vector<int>> vvgpu;
  int num_local_gpus = device_list.size();
  int num_total_gpus = num_procs * num_local_gpus;

  size_t batch_size_per_node = batch_size * num_local_gpus;
  size_t batch_size_per_iter = batch_size * num_total_gpus;
  size_t num_batches = (num_total_samples + batch_size_per_iter - 1) / batch_size_per_iter;

  size_t last_batch_iter = num_total_samples - (num_batches - 1) * batch_size_per_iter;
  size_t last_batch_gpu = last_batch_iter > rank * batch_size_per_node
                              ? last_batch_iter - rank * batch_size_per_node
                              : 0;

  size_t num_node_samples =
      (num_batches - 1) * batch_size_per_node + std::min(last_batch_gpu, batch_size_per_node);

  // if there are multi-node, we assume each node has the same gpu device_list
  for (int i = 0; i < num_procs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 424242);

  // Create AverageLoss metric
  auto metric = std::make_unique<metrics::AverageLoss<T>>(resource_manager);

  // Setup the containers
  std::vector<size_t> dims = {1, batch_size};

  // std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs(num_local_gpus);
  std::vector<core23::Tensor> loss_tensors(num_local_gpus);
  std::vector<metrics::Core23RawMetricMap> metric_maps(num_local_gpus);

  for (int i = 0; i < num_local_gpus; i++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(i)->get_device_id());
    loss_tensors[i] = core23::Tensor(core23::TensorParams()
                                         .data_type(core23::ToScalarType<float>::value)
                                         .shape({1, 1})
                                         .buffer_params(blobs_buffer_params));

    metric_maps[i] = {{metrics::RawType::Loss, loss_tensors[i]}};
  }

  std::vector<float> h_labels(num_node_samples);
  std::vector<T> h_scores(num_node_samples);
  gen(h_labels, h_scores, rank + num_procs * execution_number);
  execution_number++;

  float gpu_result, ref_result;
  for (size_t eval = 0; eval < num_evals; eval++) {
    size_t num_processed = 0;
    ref_result = 0.0;
    for (size_t batch = 0; batch < num_batches; batch++) {
      // Populate device tensors
      metric->set_current_batch_size(
          std::min(batch_size_per_iter, num_total_samples - num_processed));

      for (int i = 0; i < num_local_gpus; i++) {
        CudaDeviceContext context(resource_manager->get_local_gpu(i)->get_device_id());
        size_t start =
            std::min(batch * num_local_gpus * batch_size + i * batch_size, num_node_samples);
        size_t count =
            std::min(batch * num_local_gpus * batch_size + (i + 1) * batch_size, num_node_samples) -
            start;
        auto stream = resource_manager->get_local_gpu(i)->get_stream();

        // quickly compute the log loss per batch
        std::vector<long double> residuals = std::vector<long double>(count, 0.0);
        for (size_t c = 0; c < count; ++c) {
          long double score = static_cast<long double>(h_scores[c + start]);
          long double cutoff = 1e-3;
          if (h_labels[c + start] == 1) {
            residuals[c] = -std::log(std::max(score, cutoff)) / count;
          } else {
            residuals[c] = -std::log(std::max(1.0 - score, cutoff)) / count;
          }
        }

        std::sort(residuals.begin(), residuals.end());
        long double reducer_loss =
            std::reduce(residuals.begin(), residuals.end(), 0.0, std::plus<long double>());
        float h_loss = static_cast<float>(reducer_loss);
        if (num_local_gpus) {
          h_loss = h_loss / num_local_gpus;
        }

        HCTR_LIB_THROW(cudaMemcpyAsync(loss_tensors[i].data(), &h_loss, sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        int n_nets = 1;
        ref_result += h_loss / n_nets / num_procs;

        metric->local_reduce(i, metric_maps[i]);
      }
      num_processed += batch_size_per_iter;
      metric->global_reduce(1);
    }
    gpu_result = metric->finalize_metric();
    ref_result /= num_batches;
  }

  // float ref_result = sklearn_averageloss(num_total_samples, h_labels, h_scores);
  ASSERT_NEAR(gpu_result, ref_result, eps);
}

class MPIEnvironment : public ::testing::Environment {
 protected:
  virtual void SetUp() { test::mpi_init(); }
  virtual void TearDown() { test::mpi_finalize(); }
  virtual ~MPIEnvironment(){};
};

}  // namespace

::testing::Environment* const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

TEST(averageloss_test, fp32_1gpu) { averageloss_test<float>({0}, 10, 200, gen_random<float>); }
TEST(averageloss_test, fp32_1gpu_odd) { averageloss_test<float>({0}, 10, 182, gen_random<float>); }
TEST(averageloss_test, fp32_2gpu) { averageloss_test<float>({0, 1}, 10, 440, gen_random<float>); }
TEST(averageloss_test, fp32_2gpu_odd) {
  averageloss_test<float>({0, 1}, 10, 443, gen_random<float>);
}
TEST(averageloss_test, fp32_2_random_gpu) {
  averageloss_test<float>({3, 5}, 12, 2341, gen_random<float>);
}
TEST(averageloss_test, fp32_4gpu) {
  averageloss_test<float>({0, 1, 2, 3}, 5000, 22 * 5000 + 42, gen_random<float>);
}
TEST(averageloss_test, fp32_4gpu_same) {
  averageloss_test<float>({0, 1, 2, 3}, 12, 154, gen_same<float>);
}
TEST(averageloss_test, fp32_4gpu_same_large) {
  averageloss_test<float>({0, 1, 2, 3}, 1312, 45155, gen_same<float>);
}
TEST(averageloss_test, fp32_4gpu_multi) {
  averageloss_test<float>({0, 1, 2, 3}, 4143, 94622, gen_multilobe<float>);
}
TEST(averageloss_test, fp32_8gpu) {
  averageloss_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, 4231, 891373, gen_random<float>, 2);
}
TEST(averageloss_test, fp32_8gpu_correct) {
  averageloss_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, 5423, 874345, gen_correct<float>);
}
TEST(averageloss_test, fp32_8gpu_wrong) {
  averageloss_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, 5423, 874345, gen_wrong<float>);
}
