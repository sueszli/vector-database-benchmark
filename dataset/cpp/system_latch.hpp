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

#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdint>

namespace HugeCTR {

class SystemLatch {
  using value_type = uint32_t;

 public:
  SystemLatch() : SystemLatch(1) {}

  SystemLatch(value_type expected) : expected_(expected) {
    cudaMallocManaged(&latch_, sizeof(value_type));
    cudaMemAdvise(latch_, sizeof(value_type), cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    *latch_ = expected;
  }

  SystemLatch(const SystemLatch& other) : SystemLatch(other.expected_) {}

  SystemLatch& operator=(const SystemLatch& other) {
    expected_ = other.expected_;
    *latch_ = expected_;
    return *this;
  }

  ~SystemLatch() { cudaFree(latch_); }

  void reset(size_t expected = 0) {
    if (expected > 0) {
      expected_ = expected;
    }
    *latch_ = expected_;
  }

  void wait() const {
    while (*((volatile value_type*)latch_) > 0) {
      // do nothing
    }
  }

  void count_down(value_type n = 1) { __sync_fetch_and_sub(latch_, n); }

  static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* user_data) {
    value_type* latch = reinterpret_cast<value_type*>(user_data);
    __sync_fetch_and_sub(latch, 1);
  }

  void device_count_down(cudaStream_t stream, value_type n = 1, bool from_graph = false);

 private:
  value_type expected_;
  value_type* latch_;
};

}  // namespace HugeCTR