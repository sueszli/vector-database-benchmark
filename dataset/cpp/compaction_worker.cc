/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <eventql/util/stdtypes.h>
#include <eventql/util/logging.h>
#include <eventql/util/io/fileutil.h>
#include <eventql/util/io/mmappedfile.h>
#include <eventql/util/protobuf/msg.h>
#include <eventql/util/wallclock.h>
#include <eventql/util/application.h>
#include <eventql/db/compaction_worker.h>
#include <eventql/db/partition_writer.h>
#include <eventql/util/protobuf/MessageDecoder.h>
#include <eventql/io/cstable/RecordShredder.h>
#include <eventql/io/cstable/cstable_writer.h>

#include "eventql/eventql.h"

namespace eventql {

CompactionWorker::CompactionWorker(
    PartitionMap* pmap,
    size_t nthreads) :
    pmap_(pmap),
    nthreads_(nthreads),
    queue_([] (
        const Pair<uint64_t, RefPtr<Partition>>& a,
        const Pair<uint64_t, RefPtr<Partition>>& b) {
      return a.first < b.first;
    }),
    running_(false) {
  pmap->subscribeToPartitionChanges([this] (
      RefPtr<eventql::PartitionChangeNotification> change) {
    enqueuePartition(change->partition);
  });

  start();
}

CompactionWorker::~CompactionWorker() {
  stop();
}

void CompactionWorker::enqueuePartition(
    RefPtr<Partition> partition,
    bool immediate /* = false */) {
  if (immediate) {
    startImmediateCompaction(partition);
  } else {
    std::unique_lock<std::mutex> lk(mutex_);
    enqueuePartitionWithLock(partition);
  }
}

void CompactionWorker::startImmediateCompaction(RefPtr<Partition> partition) {
  auto uuid = partition->uuid();

  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (immediate_set_.count(uuid) > 0) {
      return;
    }

    if (immediate_set_.size() > kImmediateCompactionMaxThreads) {
      logWarning(
          "evqld",
          "maximum number of immediate compaction threads reached -- "
          "kImmediateCompactionMaxThreads or num_compaction_threads too low?");

      enqueuePartitionWithLock(partition);
      return;
    }

    immediate_set_.emplace(uuid);
  }

  auto thread = std::thread([this, partition, uuid] {
    Application::setCurrentThreadName("evqld-compaction-immediate");
    compactPartition(partition);

    std::unique_lock<std::mutex> lk(mutex_);
    immediate_set_.erase(uuid);
  });

  thread.detach();
}

void CompactionWorker::enqueuePartitionWithLock(
    RefPtr<Partition> partition) {
  auto interval = partition->getTable()->commitInterval();

  auto uuid = partition->uuid();
  if (waitset_.count(uuid) > 0) {
    return;
  }

  queue_.emplace(
      WallClock::unixMicros() + interval.microseconds(),
      partition);

  waitset_.emplace(uuid);
  cv_.notify_all();
  evqld_stats()->compaction_queue_length.set(queue_.size());
}

void CompactionWorker::start() {
  running_ = true;

  for (size_t i = 0; i < nthreads_; ++i) {
    threads_.emplace_back(std::bind(&CompactionWorker::work, this));
  }
}

void CompactionWorker::stop() {
  if (!running_) {
    return;
  }

  running_ = false;
  cv_.notify_all();

  for (auto& t : threads_) {
    t.join();
  }
}

void CompactionWorker::work() {
  Application::setCurrentThreadName("evqld-compaction");

  std::unique_lock<std::mutex> lk(mutex_);

  while (running_) {
    if (queue_.size() == 0) {
      cv_.wait(lk);
    }

    if (queue_.size() == 0) {
      continue;
    }

    auto now = WallClock::unixMicros();
    if (now < queue_.begin()->first) {
      cv_.wait_for(
          lk,
          std::chrono::microseconds(queue_.begin()->first - now));

      continue;
    }

    auto partition = queue_.begin()->second;
    queue_.erase(queue_.begin());

    bool success = true;
    {
      lk.unlock();
      success = compactPartition(partition);
      lk.lock();
    }

    if (success) {
      waitset_.erase(partition->uuid());

      if (partition->getWriter()->needsCompaction()) {
        enqueuePartitionWithLock(partition);
      }
    } else {
      auto delay = 30 * kMicrosPerSecond; // FIXPAUL increasing delay..
      queue_.emplace(now + delay, partition);
    }

    evqld_stats()->compaction_queue_length.set(queue_.size());
  }
}

bool CompactionWorker::compactPartition(RefPtr<Partition> partition) try {
  auto writer = partition->getWriter();
  if (writer->compact()) {
    auto change = mkRef(new PartitionChangeNotification());
    change->partition = partition;
    pmap_->publishPartitionChange(change);
  }

  return true;
} catch (const StandardException& e) {
  auto snap = partition->getSnapshot();

  logCritical(
      "tsdb",
      e,
      "CompactionWorker error for partition $0/$1/$2",
      snap->state.tsdb_namespace(),
      snap->state.table_key(),
      snap->key.toString());

  return false;
}

} // namespace eventql
