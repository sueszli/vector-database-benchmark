// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "threadpoolimpl.h"
#include "threadimpl.h"
#include <vespa/vespalib/util/exceptions.h>
#include <cassert>
#include <thread>

#include <vespa/log/log.h>
LOG_SETUP(".storageframework.thread_pool_impl");

using namespace std::chrono_literals;
using vespalib::IllegalStateException;

namespace storage::framework::defaultimplementation {

ThreadPoolImpl::ThreadPoolImpl(Clock& clock)
    : _clock(clock),
      _stopping(false)
{ }

ThreadPoolImpl::~ThreadPoolImpl()
{
    {
        std::lock_guard lock(_threadVectorLock);
        _stopping = true;
        for (ThreadImpl * thread : _threads) {
            thread->interrupt();
        }
        for (ThreadImpl * thread : _threads) {
            thread->join();
        }
    }
    for (uint32_t i=0; true; i+=10) {
        {
            std::lock_guard lock(_threadVectorLock);
            if (_threads.empty()) break;
        }
        if (i > 1000) {
            fprintf(stderr, "Failed to kill thread pool. Threads won't die. (And if allowing thread pool object"
                            " to be deleted this will create a segfault later)\n");
            LOG_ABORT("should not be reached");
        }
        std::this_thread::sleep_for(10ms);
    }
}

Thread::UP
ThreadPoolImpl::startThread(Runnable& runnable, vespalib::stringref id, vespalib::duration waitTime,
                            vespalib::duration maxProcessTime, int ticksBeforeWait,
                            std::optional<vespalib::CpuUsage::Category> cpu_category)
{
    std::lock_guard lock(_threadVectorLock);
    assert(!_stopping);
    auto thread = std::make_unique<ThreadImpl>(*this, runnable, id, waitTime, maxProcessTime, ticksBeforeWait, cpu_category);
    _threads.push_back(thread.get());
    return thread;
}

void
ThreadPoolImpl::visitThreads(ThreadVisitor& visitor) const
{
    std::lock_guard lock(_threadVectorLock);
    for (const ThreadImpl * thread : _threads) {
        visitor.visitThread(*thread);
    }
}

void
ThreadPoolImpl::unregisterThread(ThreadImpl& t)
{
    std::lock_guard lock(_threadVectorLock);
    std::vector<ThreadImpl*> threads;
    threads.reserve(_threads.size());
    for (ThreadImpl * thread : _threads) {
        if (thread != &t) {
            threads.push_back(thread);
        }
    }
    _threads.swap(threads);
}

}
