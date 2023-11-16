/*
 * Copyright 2014-2023 Real Logic Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "concurrent/aeron_atomic.h"

extern bool aeron_cas_int64(volatile int64_t *dst, int64_t expected, int64_t desired);

extern bool aeron_cas_uint64(volatile uint64_t *dst, uint64_t expected, uint64_t desired);

extern bool aeron_cas_int32(volatile int32_t *dst, int32_t expected, int32_t desired);

extern void aeron_acquire(void);

extern void aeron_release(void);
