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

#include "util/aeron_bitutil.h"

extern uint8_t *aeron_cache_line_align_buffer(uint8_t *buffer);
extern int aeron_number_of_trailing_zeroes(int32_t value);
extern int aeron_number_of_trailing_zeroes_u64(uint64_t value);
extern int aeron_number_of_leading_zeroes(int32_t value);
extern int32_t aeron_find_next_power_of_two(int32_t value);
