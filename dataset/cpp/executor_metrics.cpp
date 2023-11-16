// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "executor_metrics.h"

namespace proton {

void
ExecutorMetrics::update(const vespalib::ExecutorStats &stats)
{
    accepted.inc(stats.acceptedTasks);
    rejected.inc(stats.rejectedTasks);
    wakeupCount.inc(stats.wakeupCount);
    util.set(stats.getUtil());
    saturation.set(stats.get_saturation());
    const auto & qSize = stats.queueSize;
    queueSize.addValueBatch(qSize.average(), qSize.count(), qSize.min(), qSize.max());
}

ExecutorMetrics::ExecutorMetrics(const std::string &name, metrics::MetricSet *parent)
    : metrics::MetricSet(name, {}, "Instance specific thread executor metrics", parent),
      accepted("accepted", {}, "Number of accepted tasks", this),
      rejected("rejected", {}, "Number of rejected tasks", this),
      wakeupCount("wakeups", {}, "Number of times a worker thread has been woken up", this),
      util("utilization", {}, "Ratio of time the worker threads has been active", this),
      saturation("saturation", {}, "Ratio indicating how saturated the worker threads has been. "
                                   "For most executors this ratio is equal to utilization, but for others (e.g SequencedTaskExecutor) "
                                   " a higher saturation than utilization indicates a bottleneck in a subset of the worker threads.", this),
      queueSize("queuesize", {}, "Size of task queue", this)
{
}

ExecutorMetrics::~ExecutorMetrics() = default;

} // namespace proton
