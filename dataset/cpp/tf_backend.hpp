/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <nccl.h>

#include "HugeCTR/core/core.hpp"
#include "lookup/impl/core_impl/gpu_resource_impl.hpp"

namespace tf_internal {

class TFCoreResourceManager : public core::CoreResourceManager {
 public:
  TFCoreResourceManager(OpKernelContext *ctx, int device_id, int local_rank, int num_rank,
                        int id_in_local_rank, int num_gpu_per_rank,
                        HugeCTR::core23::KernelParams kernel_params)
      : core::CoreResourceManager(kernel_params),
        ctx_(ctx),
        device_id_(device_id),
        local_rank_(local_rank),
        num_rank_(num_rank),
        id_in_local_rank_(id_in_local_rank),
        num_gpu_per_rank_(num_gpu_per_rank),
        gpu_resource_(std::make_shared<GPUResource>(ctx)) {}

  std::shared_ptr<core::GPUResourceBase> get_local_gpu() override { return gpu_resource_; }

  const ncclComm_t &get_nccl() const override { return comm_; }  // TODO

  int get_local_gpu_id() const override {
    // return ctx_->device()->tensorflow_gpu_device_info()->gpu_id;
    return id_in_local_rank_;
  }

  int get_global_gpu_id() const override {
    return local_rank_ * num_gpu_per_rank_ + get_local_gpu_id();
  }
#ifdef TF_GE_210
  int get_device_id() const override {
    return ctx_->device()->tensorflow_accelerator_device_info()->gpu_id;
  }
#endif

#ifdef TF_LESS_210
  int get_device_id() const override {
    return ctx_->device()->tensorflow_gpu_device_info()->gpu_id;
  }
#endif

  size_t get_local_gpu_count() const override { return num_gpu_per_rank_; }

  size_t get_global_gpu_count() const override { return num_gpu_per_rank_ * num_rank_; }

  int get_gpu_global_id_from_local_id(int local_id) const override {
    return local_rank_ * num_gpu_per_rank_ + local_id;
  }

  int get_gpu_local_id_from_global_id(int global_id) const override {
    return global_id % num_gpu_per_rank_;
  }

 private:
  OpKernelContext *ctx_;
  int device_id_;
  // Each process corresponds to a unique rank id, i.e., local_rank_
  int local_rank_;
  int num_rank_;
  // Each process may control more than one gpu
  int id_in_local_rank_;
  int num_gpu_per_rank_;
  std::shared_ptr<core::GPUResourceBase> gpu_resource_;
  ncclComm_t comm_;
};

}  // namespace tf_internal
