#pragma once

#include <functional>

#include "abstract_task.hpp"

namespace hyrise {

/**
 * A general purpose Task for any kind of work (i.e. anything that fits into a void()-function) that can be
 * parallelized.
 *
 * Usage example:
 *
 *
 * std::atomic_uint32_t c{0}
 *
 * auto job0 = std::make_shared<JobTask>([c]() { c++; });
 * job0->schedule();
 *
 * auto job1 = std::make_shared<JobTask>([c]() { c++; });
 * job1->schedule();
 *
 * AbstractTask::wait_for_tasks({job0, job1});
 *
 * // c == 2 now
 *
 */
class JobTask : public AbstractTask {
 public:
  explicit JobTask(const std::function<void()>& fn, SchedulePriority priority = SchedulePriority::Default,
                   bool stealable = true)
      : AbstractTask{priority, stealable}, _fn{fn} {}

 protected:
  void _on_execute() override;

 private:
  std::function<void()> _fn;
};
}  // namespace hyrise
