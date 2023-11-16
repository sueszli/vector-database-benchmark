// Copyright (c) YugaByte, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.

#include "yb/server/pprof-path-handlers_util.h"

#include <cstdint>
#include <iomanip>
#include <string>
#include <utility>
#include <unordered_map>

#if YB_ABSL_ENABLED
#include "absl/debugging/symbolize.h"
#endif

#include "yb/util/logging.h"

#include "yb/gutil/strings/numbers.h"
#include "yb/util/format.h"
#include "yb/util/monotime.h"
#include "yb/util/symbolize.h"
#include "yb/util/tcmalloc_util.h"
#include "yb/util/url-coding.h"

using std::vector;

namespace yb {

#if YB_TCMALLOC_ENABLED
namespace {
void SortSamplesByOrder(vector<Sample>* samples, SampleOrder order) {
  std::sort(
      samples->begin(),
      samples->end(),
      [order](const Sample& a, const Sample& b) {
        if (order == SampleOrder::kSampledBytes) {
          return a.second.sampled_allocated_bytes > b.second.sampled_allocated_bytes;
        } else if (order == SampleOrder::kEstimatedBytes &&
                   a.second.estimated_bytes &&
                   b.second.estimated_bytes) {
          return *a.second.estimated_bytes > *b.second.estimated_bytes;
        }
        return a.second.sampled_count > b.second.sampled_count;
      });
}
} // namespace
#endif // YB_TCMALLOC_ENABLED

#if YB_GPERFTOOLS_TCMALLOC || YB_GOOGLE_TCMALLOC
namespace {
bool Symbolize(void *pc, char *out, int out_size) {
#if YB_ABSL_ENABLED
  return absl::Symbolize(pc, out, out_size);
#else
  return GlogSymbolize(pc, out, out_size);
#endif
}
}  // namespace
#endif

#if YB_GOOGLE_TCMALLOC
tcmalloc::Profile GetAllocationProfile(int seconds, int64_t sample_freq_bytes) {
  auto prev_sample_rate = tcmalloc::MallocExtension::GetProfileSamplingRate();
  tcmalloc::MallocExtension::SetProfileSamplingRate(sample_freq_bytes);
  tcmalloc::MallocExtension::AllocationProfilingToken token;
  token = tcmalloc::MallocExtension::StartLifetimeProfiling(/* seed_with_live_allocs= */ false);

  LOG(INFO) << Format("Sleeping for $0 seconds while profile is collected.", seconds);
  SleepFor(MonoDelta::FromSeconds(seconds));
  tcmalloc::MallocExtension::SetProfileSamplingRate(prev_sample_rate);
  return std::move(token).Stop();
}

tcmalloc::Profile GetHeapSnapshot(HeapSnapshotType snapshot_type) {
  if (snapshot_type == HeapSnapshotType::kPeakHeap) {
    return tcmalloc::MallocExtension::SnapshotCurrent(tcmalloc::ProfileType::kPeakHeap);
  } else {
    return tcmalloc::MallocExtension::SnapshotCurrent(tcmalloc::ProfileType::kHeap);
  }
}

vector<Sample> AggregateAndSortProfile(
    const tcmalloc::Profile& profile, bool only_growth, SampleOrder order) {
  LOG(INFO) << "Analyzing TCMalloc sampling profile";
  int failed_symbolizations = 0;
  std::unordered_map<std::string, SampleInfo> samples_map;

  profile.Iterate([&](const tcmalloc::Profile::Sample& sample) {
    // Deallocation samples are the same as the allocation samples, except with a negative
    // sample.count < 0 and the deallocation stack. Skip since we are not currently interested in
    // printing the deallocation stack.
    if (sample.count <= 0) {
      return;
    }

    // If we only want growth, exclude samples for which we saw a deallocation event.
    // "Censored" means we observed an allocation but not a deallocation. (Deallocation-only events
    // are not reported).
    if (only_growth && !sample.is_censored) {
      return;
    }

    std::stringstream sstream;
    // 256 is arbitrary. Symbolize will return false if the symbol is longer than that.
    char buf[256];
    for (int64_t i = 0; i < sample.depth; ++i) {
      if (Symbolize(sample.stack[i], buf, sizeof(buf))) {
        sstream << buf << std::endl;
      } else {
        ++failed_symbolizations;
        sstream << "Failed to symbolize" << std::endl;
      }
    }
    std::string stack = sstream.str();

    // Update the corresponding call stack entry with this sample's information.
    auto& entry = samples_map[stack];
    entry.sampled_allocated_bytes += sample.allocated_size;
    ++entry.sampled_count;
    entry.estimated_bytes = entry.estimated_bytes.value_or(0) + sample.sum;
    entry.estimated_count = entry.estimated_count.value_or(0) + sample.count;

    VLOG(1) << "Sampled stack: " << stack
            << ", sum: " << sample.sum
            << ", count: " << sample.count
            << ", requested_size: " << sample.requested_size
            << ", allocated_size: " << sample.allocated_size
            << ", is_censored: " << sample.is_censored
            << ", avg_lifetime: " << sample.avg_lifetime
            << ", allocator_deallocator_cpu_matched: "
            << sample.allocator_deallocator_cpu_matched.value_or("N/A");
  });
  if (failed_symbolizations > 0) {
    LOG(WARNING) << Format("Failed to symbolize $0 symbols", failed_symbolizations);
  }

  std::vector<Sample> samples_vec;
  samples_vec.reserve(samples_map.size());
  for (auto& entry : samples_map) {
    samples_vec.push_back(std::move(entry));
  }
  SortSamplesByOrder(&samples_vec, order);
  return samples_vec;
}

#endif // YB_GOOGLE_TCMALLOC

#if YB_GPERFTOOLS_TCMALLOC
namespace {
// From gperftools/src/malloc_extension.cc.
uintptr_t GetSampleCount(void** entry) {
  return reinterpret_cast<uintptr_t>(entry[0]);
}
uintptr_t GetSampleSize(void** entry) {
  return reinterpret_cast<uintptr_t>(entry[1]);
}
uintptr_t GetSampleDepth(void** entry) {
  return reinterpret_cast<uintptr_t>(entry[2]);
}
void* GetSampleProgramCounter(void** entry, uintptr_t i) {
  return entry[3 + i];
}
} // namespace

std::vector<Sample> GetAggregateAndSortHeapSnapshot(SampleOrder order) {
  if (order != SampleOrder::kSampledBytes && order != SampleOrder::kSampledCount) {
    LOG(WARNING) << Format("Invalid sample order $0 used with gperftools tcmalloc", order);
    return {};
  }

  int sample_period;
  void** samples = MallocExtension::instance()->ReadStackTraces(&sample_period);

  int failed_symbolizations = 0;
  std::unordered_map<std::string, SampleInfo> samples_map;

  // Samples are stored in a flattened array, where each sample is
  // [count, size, depth, stackframe 0, stackframe 1,...].
  // The end of the array is marked by a count of 0.
  for (void** sample = samples; GetSampleCount(sample) != 0; sample += 3 + GetSampleDepth(sample)) {
    std::stringstream sstream;
    // 256 is arbitrary. Symbolize will return false if the symbol is longer than that.
    char buf[256];
    for (uintptr_t i = 0; i < GetSampleDepth(sample); ++i) {
      if (Symbolize(GetSampleProgramCounter(sample, i), buf, sizeof(buf))) {
        sstream << buf << std::endl;
      } else {
        ++failed_symbolizations;
        sstream << "Failed to symbolize" << std::endl;
      }
    }
    auto stack = sstream.str();

    auto& entry = samples_map[stack];
    entry.sampled_allocated_bytes += GetSampleSize(sample);
    entry.sampled_count += GetSampleCount(sample);

    VLOG(1) << "Sampled stack: " << stack
            << ", size: " << GetSampleSize(sample)
            << ", count: " << GetSampleCount(sample);
  }
  if (failed_symbolizations > 0) {
    LOG(WARNING) << Format("Failed to symbolize $0 symbols", failed_symbolizations);
  }

  std::vector<Sample> samples_vec;
  samples_vec.reserve(samples_map.size());
  for (auto& entry : samples_map) {
    samples_vec.push_back(std::move(entry));
  }
  SortSamplesByOrder(&samples_vec, order);
  delete[] samples;
  return samples_vec;
}

#endif // YB_GPERFTOOLS_TCMALLOC

namespace {
  std::string FormatNumericTableRow(const std::string& value) {
    return Format("<td align=\"right\">$0</td>", value);
  }

  std::string FormatNumericTableRow(int64_t value) {
    return FormatNumericTableRow(SimpleItoaWithCommas(value));
  }

  std::string FormatNumericTableRow(std::optional<int64_t> value) {
    if (value) {
      return FormatNumericTableRow(*value);
    }
    return FormatNumericTableRow("N/A");
  }
} // namespace

void GenerateTable(std::stringstream* output, const std::vector<Sample>& samples,
    const std::string& title, size_t max_call_stacks, SampleOrder order) {
  // Generate the output table.
  (*output) << std::fixed;
  (*output) << std::setprecision(2);
  (*output) << Format("<b>Top $0 call stacks for:</b> $1<br>\n", max_call_stacks, title);
  if (samples.size() > max_call_stacks) {
    (*output) << Format("$0 call stacks omitted<br>\n", samples.size() - max_call_stacks);
  }
  (*output) << Format("<b>Ordering call stacks by:</b> $0<br>\n", order);
  (*output) << Format(
      "<b>Current sampling frequency:</b> $0 bytes (on average)<br>\n",
      GetTCMallocSamplingFrequency());
  (*output) << Format("Values shown below are for allocations still in use "
      "(i.e., objects that have been deallocated are not included)<br>\n");
  (*output) << "<p>\n";
  (*output) << "<table style=\"border-collapse: collapse\" border=1>\n";
  (*output) << "<style>td, th { padding: 5px; }</style>";
  (*output) << "<tr>\n";
  (*output) << "<th>Estimated Bytes</th>\n";
  (*output) << "<th>Estimated Count</th>\n";
  (*output) << "<th>Avg Bytes Per Allocation</th>\n";
  (*output) << "<th>Sampled Bytes</th>\n";
  (*output) << "<th>Sampled Count</th>\n";
  (*output) << "<th>Call Stack</th>\n";
  (*output) << "</tr>\n";

  for (size_t i = 0; i < std::min(max_call_stacks, samples.size()); ++i) {
    const auto& entry = samples.at(i);
    (*output) << "<tr>";

    (*output) << FormatNumericTableRow(entry.second.estimated_bytes);

    (*output) << FormatNumericTableRow(entry.second.estimated_count);

    std::optional<int64_t> avg_bytes;
    if (entry.second.sampled_count > 0) {
      avg_bytes = std::round(
          static_cast<double>(entry.second.sampled_allocated_bytes) / entry.second.sampled_count);
    }
    (*output) << FormatNumericTableRow(avg_bytes);

    (*output) << FormatNumericTableRow(entry.second.sampled_allocated_bytes);

    (*output) << FormatNumericTableRow(entry.second.sampled_count);

    // Call stack.
    (*output) << Format("<td><pre>$0</pre></td>", EscapeForHtmlToString(entry.first));

    (*output) << "</tr>";
  }
  (*output) << "</table>";
}

} // namespace yb
