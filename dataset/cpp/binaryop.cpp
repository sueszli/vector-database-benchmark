/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <vector>

// This set of benchmarks is designed to be a comparison for the AST benchmarks

enum class TreeType {
  IMBALANCED_LEFT  // All operator expressions have a left child operator expression and a right
                   // child column reference
};

template <typename key_type, TreeType tree_type, bool reuse_columns>
class BINARYOP : public cudf::benchmark {};

template <typename key_type, TreeType tree_type, bool reuse_columns>
static void BM_binaryop_transform(benchmark::State& state)
{
  auto const table_size{static_cast<cudf::size_type>(state.range(0))};
  auto const tree_levels{static_cast<cudf::size_type>(state.range(1))};

  // Create table data
  auto const n_cols       = reuse_columns ? 1 : tree_levels + 1;
  auto const source_table = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<key_type>()}, n_cols), row_count{table_size});
  cudf::table_view table{*source_table};

  // Execute benchmark
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    // Execute tree that chains additions like (((a + b) + c) + d)
    auto const op               = cudf::binary_operator::ADD;
    auto const result_data_type = cudf::data_type(cudf::type_to_id<key_type>());
    if (reuse_columns) {
      auto result = cudf::binary_operation(table.column(0), table.column(0), op, result_data_type);
      for (cudf::size_type i = 0; i < tree_levels - 1; i++) {
        result = cudf::binary_operation(result->view(), table.column(0), op, result_data_type);
      }
    } else {
      auto result = cudf::binary_operation(table.column(0), table.column(1), op, result_data_type);
      std::for_each(std::next(table.begin(), 2), table.end(), [&](auto const& col) {
        result = cudf::binary_operation(result->view(), col, op, result_data_type);
      });
    }
  }

  // Use the number of bytes read from global memory
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          (tree_levels + 1) * sizeof(key_type));
}

#define BINARYOP_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type, reuse_columns) \
  BENCHMARK_TEMPLATE_DEFINE_F(BINARYOP, name, key_type, tree_type, reuse_columns)     \
  (::benchmark::State & st) { BM_binaryop_transform<key_type, tree_type, reuse_columns>(st); }

BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_int32_imbalanced_unique,
                                    int32_t,
                                    TreeType::IMBALANCED_LEFT,
                                    false);
BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_int32_imbalanced_reuse,
                                    int32_t,
                                    TreeType::IMBALANCED_LEFT,
                                    true);
BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_double_imbalanced_unique,
                                    double,
                                    TreeType::IMBALANCED_LEFT,
                                    false);

static void CustomRanges(benchmark::internal::Benchmark* b)
{
  auto row_counts       = std::vector<cudf::size_type>{100'000, 1'000'000, 10'000'000, 100'000'000};
  auto operation_counts = std::vector<cudf::size_type>{1, 2, 5, 10};
  for (auto const& row_count : row_counts) {
    for (auto const& operation_count : operation_counts) {
      b->Args({row_count, operation_count});
    }
  }
}

BENCHMARK_REGISTER_F(BINARYOP, binaryop_int32_imbalanced_unique)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(BINARYOP, binaryop_int32_imbalanced_reuse)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(BINARYOP, binaryop_double_imbalanced_unique)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
