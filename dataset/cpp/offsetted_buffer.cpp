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

#include <core23/buffer.hpp>
#include <core23/offsetted_buffer.hpp>

namespace HugeCTR {

namespace core23 {

OffsettedBuffer::~OffsettedBuffer() {}

void* OffsettedBuffer::data() const {
  if (buffer_ == nullptr) {
    return nullptr;
  }

  if (!initialized_) {
    buffer_->allocate();
    // At this point, Buffer makes this OffsetBuffer initialized and have the correct offset.
  }
  return buffer_->data(offset_, {});
}

}  // namespace core23

}  // namespace HugeCTR
