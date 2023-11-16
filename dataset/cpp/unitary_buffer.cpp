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

#include <core23/allocator.hpp>
#include <core23/buffer_client.hpp>
#include <core23/details/unitary_buffer.hpp>
#include <core23/device.hpp>
#include <core23/logger.hpp>
#include <core23/offsetted_buffer.hpp>
#include <memory>

namespace HugeCTR {

namespace core23 {

UnitaryBuffer::UnitaryBuffer(const Device& device, std::unique_ptr<Allocator> allocator)
    : Buffer(device, std::move(allocator)),
      allocated_(false),
      ptr_(nullptr),
      current_offset_(0LL) {}

UnitaryBuffer::~UnitaryBuffer() {
  if (allocated_ || ptr_ != nullptr) allocator()->deallocate(ptr_);
}

size_t UnitaryBuffer::do_get_reserved_size(const std::unique_ptr<Allocator>& allocator,
                                           const ClientRequirements& client_requirements) {
  if (client_requirements.empty()) {
    return 0;
  }
  int64_t current_offset = 0;
  bool is_first = true;

  std::queue<BufferClient*> tmp_order = new_insertion_order_;
  while (!tmp_order.empty()) {
    auto client = tmp_order.front();
    auto search = client_requirements.find(client);
    if (search != client_requirements.end()) {
      auto requirements = search->second;
      int64_t alignment = requirements.alignment;
      if (is_first) {
        alignment = allocator->get_valid_alignment(alignment);
        is_first = false;
      }
      current_offset = compute_offset(current_offset, alignment);
      current_offset += requirements.num_bytes;
    }
    tmp_order.pop();
  }
  return current_offset;
}
Buffer::ClientOffsets UnitaryBuffer::do_allocate(const std::unique_ptr<Allocator>& allocator,
                                                 const ClientRequirements& client_requirements) {
  if (client_requirements.empty()) {
    HCTR_OWN_THROW(
        HugeCTR::Error_t::IllegalCall,
        "The buffer doesn't have any subscriber at all. What is the point of allocate()?");
  }

  if (allocated_) {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "The UnitaryBuffer doesn't allow the multiple allocation.");
  }

  int64_t current_offset = 0;
  bool is_first = true;
  const auto& first_stream = client_requirements.begin()->second.stream;

  ClientOffsets client_offsets;

  while (!new_insertion_order_.empty()) {
    auto client = new_insertion_order_.front();
    auto search = client_requirements.find(client);
    if (search != client_requirements.end()) {
      auto requirements = search->second;
      int64_t alignment = requirements.alignment;
      if (is_first) {
        alignment = allocator->get_valid_alignment(alignment);
        is_first = false;
      } else {
        const auto& current_stream = requirements.stream;
        if (first_stream != current_stream) {
          HCTR_LOG_S(WARNING, ROOT)
              << "A BufferClient doesn't have the same CUDAStream with the first BufferClient."
              << std::endl;
        }
      }

      current_offset = compute_offset(current_offset, alignment);
      client_offsets[client] = current_offset;
      current_offset += requirements.num_bytes;
    }
    new_insertion_order_.pop();
  }
  ptr_ = allocator->allocate(current_offset, first_stream);
  if (ptr_ == nullptr && current_offset) {
    HCTR_OWN_THROW(HugeCTR::Error_t::OutOfMemory,
                   "The UnitaryBuffer failed to allocate the memory");
  }
  allocated_ = true;
  current_offset_ = current_offset;

  return client_offsets;
}

void UnitaryBuffer::post_subscribe(const BufferClient* client, BufferRequirements requirements) {
  new_insertion_order_.push(const_cast<BufferClient*>(client));
}

int64_t UnitaryBuffer::compute_offset(int64_t offset, int64_t alignment) {
  if (alignment != 0) {
    int64_t rem = offset % alignment;
    if (rem != 0) {
      offset += alignment;
      offset -= rem;
    }
  }
  return offset;
}

}  // namespace core23

}  // namespace HugeCTR
