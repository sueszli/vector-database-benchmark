//
// Created by XingfengYang on 2020/7/17.
//

#include "kernel/semaphore.h"
#include "arm/register.h"
#include "kernel/assert.h"
#include "kernel/percpu.h"
#include "kernel/scheduler.h"
#include "kernel/thread.h"

extern Scheduler cfsScheduler;

void semaphore_default_post(Semaphore *semaphore) {
    semaphore->spinLock.operations.acquire(&semaphore->spinLock);

    atomic_inc(&semaphore->count);
    KQueueNode *queueNode = semaphore->waitQueue.operations.dequeue(&semaphore->waitQueue);
    Thread *releasedThread = getNode(queueNode, Thread, threadReadyQueue);


    KernelStatus addToScheduler = cfsScheduler.operation.addThread(&cfsScheduler, releasedThread,
                                                                   releasedThread->priority);

    DEBUG_ASSERT(addToScheduler == OK);

    semaphore->spinLock.operations.release(&semaphore->spinLock);
}

void semaphore_default_wait(Semaphore *semaphore) {
    semaphore->spinLock.operations.acquire(&semaphore->spinLock);

    if (atomic_get(&semaphore->count) > 0) {
        atomic_dec(&semaphore->count);
    } else {
        uint32_t cpuid = read_cpuid();
        PerCpu *perCpu = percpu_get(cpuid);
        Thread *currentThread = perCpu->currentThread;

        DEBUG_ASSERT(currentThread != nullptr);

        // can not get the lock, just add to lock wait list
        semaphore->waitQueue.operations.enqueue(&semaphore->waitQueue, &currentThread->threadReadyQueue);
        currentThread->threadStatus = THREAD_BLOCKED;
        // remove from schd list
        perCpu->rbTree.operations.remove(&perCpu->rbTree, &currentThread->rbNode);
        // 2. switch to the next thread in scheduler
        cfsScheduler.operation.switchNext(&cfsScheduler);
    }
    semaphore->spinLock.operations.release(&semaphore->spinLock);
}
