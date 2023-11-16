//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "CommonCommonPch.h"
#ifdef _WIN32
#include <process.h>
#endif

#include "Core/EtwTraceCore.h"

#include "Exceptions/ExceptionBase.h"
#include "Exceptions/JavascriptException.h"
#include "Exceptions/OperationAbortedException.h"
#include "Exceptions/OutOfMemoryException.h"
#include "Exceptions/StackOverflowException.h"

#include "TemplateParameter.h"
#include "DataStructures/DoublyLinkedListElement.h"
#include "DataStructures/DoublyLinkedList.h"
#include "DataStructures/DoublyLinkedListElement.inl"
#include "DataStructures/DoublyLinkedList.inl"

#include "Common/Event.h"
#include "Common/ThreadService.h"
#include "Common/Jobs.h"
#include "Common/Jobs.inl"
#include "Core/CommonMinMax.h"
#include "Memory/RecyclerWriteBarrierManager.h"
#include "Memory/XDataAllocator.h"

namespace JsUtil
{
    // -------------------------------------------------------------------------------------------------------------------------
    // Job
    // -------------------------------------------------------------------------------------------------------------------------

    Job::Job(const bool isCritical) : manager(0), isCritical(isCritical)
#if ENABLE_DEBUG_CONFIG_OPTIONS
        , failureReason(FailureReason::NotFailed)
#endif
    {
    }

    Job::Job(JobManager *const manager, const bool isCritical) : manager(manager), isCritical(isCritical)
#if ENABLE_DEBUG_CONFIG_OPTIONS
        , failureReason(FailureReason::NotFailed)
#endif
    {
        Assert(manager);
    }

    JobManager *Job::Manager() const
    {
        return manager;
    }

    bool Job::IsCritical() const
    {
        return isCritical;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // JobManager
    // -------------------------------------------------------------------------------------------------------------------------

    JobManager::JobManager(JobProcessor *const processor)
        : processor(processor), numJobsAddedToProcessor(0), isWaitable(false)
    {
        Assert(processor);
    }

    JobManager::JobManager(JobProcessor *const processor, const bool isWaitable)
        : processor(processor), numJobsAddedToProcessor(0), isWaitable(isWaitable)
    {
        Assert(processor);
    }

    JobProcessor *JobManager::Processor() const
    {
        return processor;
    }

    void JobManager::LastJobProcessed()
    {
    }

    void JobManager::ProcessorThreadSpecificCallBack(PageAllocator *pageAllocator)
    {
    }

    Job *JobManager::GetJobToProcessProactively()
    {
        return 0;
    }

    bool JobManager::ShouldProcessInForeground(const bool willWaitForJob, const unsigned int numJobsInQueue) const
    {
        return false;
    }

    void JobManager::Prioritize(JsUtil::Job *const job, const bool forceAddJobToProcessor, void* function) const
    {
    }

    void JobManager::PrioritizedButNotYetProcessed(JsUtil::Job *const job) const
    {
    }

    void JobManager::OnDecommit(ParallelThreadData *threadData)
    {
    }

    void JobManager::BeforeWaitForJob(bool) const
    {
    }

    void JobManager::AfterWaitForJob(bool) const
    {
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // WaitableJobManager
    // -------------------------------------------------------------------------------------------------------------------------

    WaitableJobManager::WaitableJobManager(JobProcessor *const processor)
        : JobManager(processor, true),
        jobBeingWaitedUpon(0),
#if ENABLE_BACKGROUND_JOB_PROCESSOR
        jobBeingWaitedUponProcessed(false),
#endif
        isWaitingForQueuedJobs(false)
#if ENABLE_BACKGROUND_JOB_PROCESSOR
        , queuedJobsProcessed(false)
#endif
    {
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // SingleJobManager
    // -------------------------------------------------------------------------------------------------------------------------

    SingleJobManager::SingleJobManager(JobProcessor *const processor, const bool isCritical)
        : JobManager(processor), job(isCritical), processed(false)
    {
        job.manager = this;
    }

    void SingleJobManager::AddJobToProcessor(const bool prioritize)
    {
        AutoOptionalCriticalSection lock(Processor()->GetCriticalSection());
        Processor()->AddJob(&job, prioritize);
    }

    void SingleJobManager::JobProcessed(JsUtil::Job *const job, const bool succeeded)
    {
        processed = true;
    }

    JsUtil::Job *SingleJobManager::GetJob(bool)
    {
        return processed ? 0 : &job;
    }

    bool SingleJobManager::WasAddedToJobProcessor(JsUtil::Job *const job) const
    {
        return true;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // WaitableSingleJobManager
    // -------------------------------------------------------------------------------------------------------------------------

    WaitableSingleJobManager::WaitableSingleJobManager(JobProcessor *const processor, const bool isCritical)
        : WaitableJobManager(processor), job(isCritical), processed(false)
    {
        job.manager = this;
    }

    void WaitableSingleJobManager::AddJobToProcessor(const bool prioritize)
    {
        AutoOptionalCriticalSection lock(Processor()->GetCriticalSection());
        Processor()->AddJob(&job, prioritize);
    }

    void WaitableSingleJobManager::WaitForJobProcessed()
    {
        Processor()->PrioritizeJobAndWait(this, false);
    }

    void WaitableSingleJobManager::JobProcessed(JsUtil::Job *const job, const bool succeeded)
    {
        processed = true;
    }

    JsUtil::Job *WaitableSingleJobManager::GetJob(bool)
    {
        return processed ? 0 : &job;
    }

    bool WaitableSingleJobManager::WasAddedToJobProcessor(JsUtil::Job *const job) const
    {
        return true;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // JobProcessor
    // -------------------------------------------------------------------------------------------------------------------------

    JobProcessor::JobProcessor(const bool processesInBackground) : processesInBackground(processesInBackground), isClosed(false)
    {
    }

    bool JobProcessor::ProcessesInBackground() const
    {
        return processesInBackground;
    }

    bool JobProcessor::IsClosed() const
    {
        return isClosed;
    }

    CriticalSection *JobProcessor::GetCriticalSection()
    {
#if ENABLE_BACKGROUND_JOB_PROCESSOR
        return processesInBackground ? static_cast<BackgroundJobProcessor *>(this)->GetCriticalSection() : 0;
#else
        return 0;
#endif
    }

    void JobProcessor::AddManager(JobManager *const manager)
    {
        Assert(manager);
        Assert(!isClosed);

        managers.LinkToEnd(manager);
    }

    bool JobProcessor::HasManager(JobManager *const manager) const
    {
        for (JobManager *curManager = managers.Head(); curManager != NULL; curManager = curManager->Next())
        {
            if (manager == curManager)
            {
                return true;
            }
        }

        return false;
    }

    template<class Fn>
    void JobProcessor::ForEachManager(Fn fn)
    {
        for (JobManager *curManager = managers.Head(); curManager != NULL; curManager = curManager->Next())
        {
            fn(curManager);
        }
    }

    void JobProcessor::PrioritizeManager(JobManager *const manager)
    {
        Assert(manager);
        Assert(!isClosed);

        managers.MoveToBeginning(manager);
        if (manager->numJobsAddedToProcessor == 0)
        {
            return;
        }

        // Move this manager's jobs to the beginning too. Find sequences of this manager's jobs backwards so that their relative
        // order remains intact after the sequences are moved.
        Job *const originalHead = jobs.Head();
        Job *lastJob = 0;
        for (Job *job = jobs.Tail(); job; job = job->Previous())
        {
            if (job->Manager() == manager)
            {
                if (!lastJob)
                    lastJob = job;
            }
            else if (lastJob)
            {
                jobs.MoveSubsequenceToBeginning(job->Next(), lastJob);
                lastJob = 0;
            }

            if (job == originalHead)
            {
                break;
            }
        }
        if (lastJob)
        {
            jobs.MoveSubsequenceToBeginning(originalHead, lastJob);
        }
    }

    void JobProcessor::AddJob(Job *const job, const bool prioritize)
    {
        // This function is called from inside the lock

        Assert(job);
        Assert(managers.Contains(job->Manager()));
        Assert(!IsClosed());

        if (job->Manager()->numJobsAddedToProcessor + 1 == 0)
            Js::Throw::OutOfMemory();  // Overflow: job counts we use are int32's.
        ++job->Manager()->numJobsAddedToProcessor;

        if (prioritize)
            jobs.LinkToBeginning(job);
        else
            jobs.LinkToEnd(job);
    }

    bool JobProcessor::RemoveJob(Job *const job)
    {
        // This function is called from inside the lock

        Assert(job);
        Assert(managers.Contains(job->Manager()));
        Assert(!IsClosed());

        jobs.Unlink(job);
        Assert(job->Manager()->numJobsAddedToProcessor != 0);
        --job->Manager()->numJobsAddedToProcessor;
        return true;
    }

    void JobProcessor::JobProcessed(JobManager *const manager, Job *const job, const bool succeeded)
    {
        Assert(manager);
        Assert(job);
        Assert(manager == job->Manager());

        BEGIN_NO_EXCEPTION
        {
            manager->JobProcessed(job, succeeded);
        }
        END_NO_EXCEPTION;
    }

    void JobProcessor::LastJobProcessed(JobManager *const manager)
    {
        Assert(manager);

        BEGIN_NO_EXCEPTION
        {
            manager->LastJobProcessed();
        }
        END_NO_EXCEPTION;
    }

    void JobProcessor::Close()
    {
        isClosed = true;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // ForegroundJobProcessor
    // -------------------------------------------------------------------------------------------------------------------------

    ForegroundJobProcessor::ForegroundJobProcessor() : JobProcessor(false)
    {
    }

    void ForegroundJobProcessor::RemoveManager(JobManager *const manager)
    {
        Assert(manager);
        // Managers must remove themselves. Hence, Close does not remove managers. So, not asserting on !IsClosed().

        managers.Unlink(manager);
        if (manager->numJobsAddedToProcessor == 0)
            return;

        // Remove this manager's jobs from the queue
        Job *firstJob = 0;
        for (Job *job = jobs.Head(); job; job = job->Next())
        {
            if (job->Manager() == manager)
            {
                if (!firstJob)
                    firstJob = job;
            }
            else if (firstJob)
            {
                jobs.UnlinkSubsequence(firstJob, job->Previous());
                for (Job *removedJob = firstJob; removedJob;)
                {
                    Job *const next = removedJob->Next();
                    Assert(!removedJob->IsCritical());
                    JobProcessed(manager, removedJob, false); // the job may be deleted during this and should not be used afterwards
                    Assert(manager->numJobsAddedToProcessor != 0);
                    --manager->numJobsAddedToProcessor;
                    removedJob = next;
                }
                firstJob = 0;
            }
        }
        if (firstJob)
        {
            jobs.UnlinkSubsequenceFromEnd(firstJob);
            for (Job *removedJob = firstJob; removedJob;)
            {
                Job *const next = removedJob->Next();
                Assert(!removedJob->IsCritical());
                JobProcessed(manager, removedJob, false); // the job may be deleted during this and should not be used afterwards
                Assert(manager->numJobsAddedToProcessor != 0);
                --manager->numJobsAddedToProcessor;
                removedJob = next;
            }
        }

        Assert(manager->numJobsAddedToProcessor == 0);
        LastJobProcessed(manager);
    }

    bool ForegroundJobProcessor::Process(Job *const job)
    {
        try
        {
            return job->Manager()->Process(job, 0);
        }
        catch (const Js::JavascriptException& err)
        {
            err.GetAndClear(); // discard exception object

            // Treat OOM or stack overflow to be a non-terminal failure. The foreground job processor processes jobs when the
            // jobs are prioritized, on the calling thread. The script would be active (at the time of this writing), so a
            // JavascriptExceptionObject would be thrown for OOM or stack overflow.
        }
        catch (Js::OperationAbortedException)
        {
            // This can happen for any reason a job needs to be aborted while executing
        }

        // Any of the above exceptions will result in the job failing. The return value of this function will cause the job
        // manager to get a JobProcessed call with succeeded = false, so that it can handle the failure appropriately.
        return false;
    }

    void ForegroundJobProcessor::Close()
    {
        if (IsClosed())
            return;

        for (Job *job = jobs.Head(); job;)
        {
            Job *const next = job->Next();
            JobManager *const manager = job->Manager();
            JobProcessed(
                manager,
                job,
                job->IsCritical() ? Process(job) : false); // the job may be deleted during this and should not be used afterwards
            Assert(manager->numJobsAddedToProcessor != 0);
            --manager->numJobsAddedToProcessor;
            if (manager->numJobsAddedToProcessor == 0)
                LastJobProcessed(manager); // the manager may be deleted during this and should not be used afterwards
            job = next;
        }
        jobs.Clear();

        JobProcessor::Close();
    }

    void ForegroundJobProcessor::AssociatePageAllocator(PageAllocator* const pageAllocator)
    {
        // Do nothing
    }

    void ForegroundJobProcessor::DissociatePageAllocator(PageAllocator* const pageAllocator)
    {
        // Do nothing
    }

// Xplat-todo: revive BackgroundJobProcessor- we need this for the JIT
#if ENABLE_BACKGROUND_JOB_PROCESSOR

    // -------------------------------------------------------------------------------------------------------------------------
    // BackgroundJobProcessor
    // -------------------------------------------------------------------------------------------------------------------------

    ParallelThreadData::ParallelThreadData(AllocationPolicyManager* policyManager) :
        threadHandle(0),
        isWaitingForJobs(false),
        canDecommit(true),
        currentJob(nullptr),
        threadStartedOrClosing(false),
        backgroundPageAllocator(policyManager, Js::Configuration::Global.flags, PageAllocatorType_BGJIT,
        (AutoSystemInfo::Data.IsLowMemoryProcess() ?
            PageAllocator::DefaultLowMaxFreePageCount :
            PageAllocator::DefaultMaxFreePageCount)),
        threadArena(nullptr),
        processor(nullptr),
        parser(nullptr),
        pse(nullptr),
        scriptContextBG(nullptr)
    {
    }

    void BackgroundJobProcessor::InitializeThreadCount()
    {
        if (CONFIG_FLAG(ForceMaxJitThreadCount))
        {
            this->maxThreadCount = CONFIG_FLAG(MaxJitThreadCount);
        }
        else if (AutoSystemInfo::Data.IsLowMemoryProcess())
        {
            // In a low-memory scenario, don't spin up multiple threads, regardless of how many cores we have.
            this->maxThreadCount = 1;
        }
        else
        {
            int processorCount = AutoSystemInfo::Data.GetNumberOfPhysicalProcessors();
            //There is 2 threads already in play, one UI (main) thread and a GC thread. So subtract 2 from processorCount to account for the same.

            this->maxThreadCount = max(1, min(processorCount - 2, CONFIG_FLAG(MaxJitThreadCount)));
        }
    }

    void BackgroundJobProcessor::InitializeParallelThreadData(AllocationPolicyManager* policyManager, bool disableParallelThreads)
    {
        if (!disableParallelThreads)
        {
            InitializeThreadCount();
        }
        else
        {
            this->maxThreadCount = 1;
        }

        Assert(this->maxThreadCount >= 1);
        this->parallelThreadData = HeapNewArrayZ(ParallelThreadData*, this->maxThreadCount);

        for (uint i = 0; i < this->maxThreadCount; i++)
        {
            this->parallelThreadData[i] = HeapNewNoThrow(ParallelThreadData, policyManager);

            if (this->parallelThreadData[i] == nullptr)
            {
                if (i == 0)
                {
                    HeapDeleteArray(this->maxThreadCount, this->parallelThreadData);
                    Js::Throw::OutOfMemory();
                }
                // At least one thread is created, continue
                break;
            }

            this->parallelThreadData[i]->processor = this;
            // Make sure to create the thread suspended so the thread handle can be assigned before the thread starts running
            auto threadHandle = PlatformAgnostic::Thread::Create(0, &StaticThreadProc,
                this->parallelThreadData[i], PlatformAgnostic::Thread::ThreadInitCreateSuspended, _u("Chakra Parallel Worker Thread"));

            if (threadHandle != PlatformAgnostic::Thread::InvalidHandle)
            {
                this->parallelThreadData[i]->threadHandle = reinterpret_cast<HANDLE>(threadHandle);
            }

            if (!this->parallelThreadData[i]->threadHandle)
            {
                HeapDelete(parallelThreadData[i]);
                parallelThreadData[i] = nullptr;
                if (i == 0)
                {
                    Js::Throw::OutOfMemory();
                }
                // At least one thread is created, continue
                break;
            }

            if (ResumeThread(this->parallelThreadData[i]->threadHandle) == static_cast<DWORD>(-1))
            {
                CloseHandle(this->parallelThreadData[i]->threadHandle);
                HeapDelete(parallelThreadData[i]);
                this->parallelThreadData[i] = nullptr;

                if (i == 0)
                {
                    Js::Throw::OutOfMemory();
                }
                // At least one thread is created, continue
                break;
            }

            this->threadCount++;

            // Wait for the thread to fully start. This is necessary because Close may be called before the thread starts and if
            // Close is called while holding the loader lock during DLL_THREAD_DETACH, the thread may be stuck waiting for the
            // loader lock for DLL_THREAD_ATTACH to start up, and Close would then end up waiting forever, causing a deadlock.
            WaitWithThreadForThreadStartedOrClosingEvent(this->parallelThreadData[i]);
            this->parallelThreadData[i]->threadStartedOrClosing.Reset(); // after this, the event will be used to wait for the thread to close

#if DBG_DUMP
            if (i < (sizeof(DebugThreadNames) / sizeof(DebugThreadNames[i])))
            {
                this->parallelThreadData[i]->backgroundPageAllocator.debugName = DebugThreadNames[i];
            }
            else
            {
                this->parallelThreadData[i]->backgroundPageAllocator.debugName = _u("BackgroundJobProcessor thread");
            }
#endif
        }

        Assert(this->threadCount >= 1);
     }

    void BackgroundJobProcessor::InitializeParallelThreadDataForThreadServiceCallBack(AllocationPolicyManager* policyManager)
    {
        //thread is provided by service callback, no need to create thread here. Currently only one thread in service callback supported.
        this->maxThreadCount = 1;
        this->parallelThreadData = HeapNewArrayZ(ParallelThreadData *, this->maxThreadCount);

        this->parallelThreadData[0] = HeapNewNoThrow(ParallelThreadData, policyManager);
        if (this->parallelThreadData[0] == nullptr)
        {
            HeapDeleteArray(this->maxThreadCount, this->parallelThreadData);
            Js::Throw::OutOfMemory();
        }
        this->parallelThreadData[0]->processor = this;
        this->parallelThreadData[0]->isWaitingForJobs = true;
#if DBG_DUMP
        this->parallelThreadData[0]->backgroundPageAllocator.debugName = _u("BackgroundJobProcessor");
#endif
        this->threadCount = 1;

        return;
    }

    BackgroundJobProcessor::BackgroundJobProcessor(AllocationPolicyManager* policyManager, JsUtil::ThreadService *threadService, bool disableParallelThreads)
        : JobProcessor(true),
        jobReady(true),
        wakeAllBackgroundThreads(false),
        numJobs(0),
        threadId(GetCurrentThreadContextId()),
        threadService(threadService),
        threadCount(0),
        maxThreadCount(0)
#if PDATA_ENABLED && defined(_WIN32)
        ,hasExtraWork(0)
#endif
    {
        if (!threadService->HasCallback())
        {
            // We don't have a thread service, so create a dedicated thread to handle background jobs.
            InitializeParallelThreadData(policyManager, disableParallelThreads);
        }
        else
        {
            InitializeParallelThreadDataForThreadServiceCallBack(policyManager);
        }
    }

    BackgroundJobProcessor::~BackgroundJobProcessor()
    {
        // This should appear to be called from the same thread from which this instance was created
        Assert(IsClosed());

        if (parallelThreadData)
        {
            for (unsigned int i = 0; i < this->threadCount; i++)
            {
                HeapDelete(parallelThreadData[i]);
            }
            HeapDeleteArray(this->maxThreadCount, parallelThreadData);
        }
    }

    void BackgroundJobProcessor::WaitWithAllThreadsForThreadStartedOrClosingEvent()
    {
        bool continueWaiting = true;
        this->IterateBackgroundThreads([&](ParallelThreadData *threadData)
        {
            if (continueWaiting)
            {
                continueWaiting = WaitWithThreadForThreadStartedOrClosingEvent(threadData);
            }
            else
            {
                //one of the thread is terminated, its sure shutdown scenario. Just reset the waitingForjobs.
                threadData->isWaitingForJobs = false;
            }
            return false;
        });
    }

    template<class Fn>
    void BackgroundJobProcessor::ForEachManager(Fn fn)
    {
        AutoCriticalSection lock(&criticalSection);
        JobProcessor::ForEachManager(fn);
    }

    //This function waits on two events jobReady or wakeAllBackgroundThreads
    //It first waits for 1sec and if it times out it will decommit the allocator and wait infinitely.
    bool BackgroundJobProcessor::WaitForJobReadyOrShutdown(ParallelThreadData *threadData)
    {
        const HANDLE handles[] = { jobReady.Handle(), wakeAllBackgroundThreads.Handle() };

        //Wait for 1 sec on jobReady and shutdownBackgroundThread events.
        unsigned int result = WaitForMultipleObjectsEx(_countof(handles), handles, false, 1000, false);

#if PDATA_ENABLED && defined(_WIN32)
        DoExtraWork();
#endif

        while (result == WAIT_TIMEOUT)
        {
            if (threadData->CanDecommit())
            {
                // If its 1sec time out decommit and wait for INFINITE
                threadData->backgroundPageAllocator.DecommitNow();
                this->ForEachManager([&](JobManager *manager){
                    manager->OnDecommit(threadData);
                });

                result = WaitForMultipleObjectsEx(_countof(handles), handles, false, INFINITE, false);
            }
            else
            {
                result = WaitForMultipleObjectsEx(_countof(handles), handles, false, 1000, false);
            }
        }

        if (!(result == WAIT_OBJECT_0 || result == WAIT_OBJECT_0 + 1))
        {
            Js::Throw::FatalInternalError();
        }

        return result == WAIT_OBJECT_0;
    }

#if PDATA_ENABLED && defined(_WIN32)
    void BackgroundJobProcessor::DoExtraWork()
    {
        while (InterlockedExchangeAdd(&hasExtraWork, 0) > 0)
        {
            DelayDeletingFunctionTable::Clear();
            Sleep(50);
        }        
    }
#endif

    bool BackgroundJobProcessor::WaitWithThreadForThreadStartedOrClosingEvent(ParallelThreadData *parallelThreadData, const unsigned int milliseconds)
    {
        return WaitWithThread(parallelThreadData, parallelThreadData->threadStartedOrClosing, milliseconds);
    }

    bool BackgroundJobProcessor::WaitWithThread(ParallelThreadData *parallelThreadData, const Event &e, const unsigned int milliseconds)
    {
        const HANDLE handles[] = { e.Handle(), parallelThreadData->threadHandle };

        // If we have a thread service, then only wait on the event, not the actual thread handle.
        DWORD handleCount = 2;
        if (threadService->HasCallback())
        {
            handleCount = 1;
        }

        const unsigned int result = WaitForMultipleObjectsEx(handleCount, handles, false, milliseconds, false);

        if (!(result == WAIT_OBJECT_0 || result == WAIT_OBJECT_0 + 1 || (result == WAIT_TIMEOUT && milliseconds != INFINITE)))
        {
            Js::Throw::FatalInternalError();
        }

        if (result == WAIT_OBJECT_0 + 1)
        {
            // Apparently, sometimes the thread dies while waiting for an event. It should only be during process shutdown but
            // we can't know because DLL_PROCESS_DETACH may not have been called yet, which is bizarre. It seems unclear why
            // this happens and this could cause unpredictable behavior since the behavior of this object is undefined if the
            // thread is killed arbitrarily, or if there are incoming calls after Close. In any case, uses of this function have
            // been ported from BackgroundCodeGenThread. For now, we assume that Close will be called eventually and set the
            // state to what it should be before Close is called.
            parallelThreadData->isWaitingForJobs = false;
        }

        return result == WAIT_OBJECT_0;

    }

    void BackgroundJobProcessor::AddManager(JobManager *const manager)
    {
        Assert(manager);

        IterateBackgroundThreads([&manager](ParallelThreadData *threadData){
            manager->ProcessorThreadSpecificCallBack(threadData->GetPageAllocator());
            return false;
        });

        AutoCriticalSection lock(&criticalSection);
        Assert(!IsClosed());

        JobProcessor::AddManager(manager);


        IndicateNewJob();
    }

    void BackgroundJobProcessor::IndicateNewJob()
    {
        Assert(criticalSection.IsLocked());

        if(NumberOfThreadsWaitingForJobs ())
        {
            if (threadService->HasCallback())
            {
                Assert(this->threadCount == 1);
                this->parallelThreadData[0]->isWaitingForJobs = false;

                // Reset the thread event, so we can wait for it on shutdown.
                this->parallelThreadData[0]->threadStartedOrClosing.Reset();

                // Submit a request to the thread service.
                bool success = threadService->Invoke(ThreadServiceCallback, this);
                if (!success)
                {
                    // The thread service denied our request.
                    // Leave the job in the queue.  If it's needed, it will be processed
                    // in-thread during PrioritizeJob.  Or alternatively, if a subsequent
                    // thread service request succeeds, this job will be processed then.
                    this->parallelThreadData[0]->isWaitingForJobs = true;
                }
            }
            else
            {
                // Signal the background thread to wake up and process jobs.
                jobReady.Set();
            }
        }
    }

    Job * BackgroundJobProcessor::GetCurrentJobOfManager(JobManager *const manager)
    {
        Assert(criticalSection.IsLocked());
        Job *currentJob = nullptr;
        this->IterateBackgroundThreads([&](ParallelThreadData* threadData)
        {
            if (!threadData->currentJob)
            {
                return false;
            }
            if (threadData->currentJob->Manager() != manager)
            {
                return false;
            }
            currentJob = threadData->currentJob;
            return true;
        }
        );
        return currentJob;
    }

    ParallelThreadData * BackgroundJobProcessor::GetThreadDataFromCurrentJob(Job* job)
    {
        Assert(criticalSection.IsLocked());
        ParallelThreadData *currentThreadData = nullptr;
        this->IterateBackgroundThreads([&](ParallelThreadData* threadData)
        {
            if (!threadData->currentJob)
            {
                return false;
            }
            if (threadData->currentJob != job)
            {
                return false;
            }
            currentThreadData = threadData;
            return true;
        }
        );
        return currentThreadData;
    }

    void BackgroundJobProcessor::RemoveManager(JobManager *const manager)
    {
        Assert(manager);

        ParallelThreadData *threadDataProcessingCurrentJob = nullptr;
        {
            AutoCriticalSection lock(&criticalSection);
            // Managers must remove themselves. Hence, Close does not remove managers. So, not asserting on !IsClosed().

            if (!HasManager(manager))
            {
                // Since this manager isn't owned by this processor, no need to remove and cleanup
                return;
            }

            managers.Unlink(manager);
            if(manager->numJobsAddedToProcessor == 0)
            {
                Assert(!GetCurrentJobOfManager(manager));
                return;
            }

            // Remove this manager's jobs from the queue
            Job *firstJob = 0;
            for(Job *job = jobs.Head(); job; job = job->Next())
            {
                if(job->Manager() == manager)
                {
                    if(!firstJob)
                        firstJob = job;
                }
                else if(firstJob)
                {
                    jobs.UnlinkSubsequence(firstJob, job->Previous());
                    for(Job *removedJob = firstJob; removedJob;)
                    {
                        Job *const next = removedJob->Next();
                        Assert(!removedJob->IsCritical());
                        Assert(numJobs != 0);
                        --numJobs;
                        JobProcessed(manager, removedJob, false); // the job may be deleted during this and should not be used afterwards
                        Assert(manager->numJobsAddedToProcessor != 0);
                        --manager->numJobsAddedToProcessor;
                        if(manager->isWaitable)
                        {
                            WaitableJobManager *const waitableManager = static_cast<WaitableJobManager *>(manager);
                            if(waitableManager->jobBeingWaitedUpon == removedJob)
                            {
                                waitableManager->jobBeingWaitedUponProcessed.Set();
                                waitableManager->jobBeingWaitedUpon = 0;
                            }
                        }
                        removedJob = next;
                    }
                    firstJob = 0;
                }
            }
            if(firstJob)
            {
                jobs.UnlinkSubsequenceFromEnd(firstJob);
                for(Job *removedJob = firstJob; removedJob;)
                {
                    Job *const next = removedJob->Next();
                    Assert(!removedJob->IsCritical());
                    Assert(numJobs != 0);
                    --numJobs;
                    JobProcessed(manager, removedJob, false); // the job may be deleted during this and should not be used afterwards
                    Assert(manager->numJobsAddedToProcessor != 0);
                    --manager->numJobsAddedToProcessor;
                    if(manager->isWaitable)
                    {
                        WaitableJobManager *const waitableManager = static_cast<WaitableJobManager *>(manager);
                        if(waitableManager->jobBeingWaitedUpon == removedJob)
                        {
                            waitableManager->jobBeingWaitedUponProcessed.Set();
                            waitableManager->jobBeingWaitedUpon = 0;
                        }
                    }
                    removedJob = next;
                }
            }

            if(manager->numJobsAddedToProcessor == 0)
            {
                LastJobProcessed(manager);
                return;
            }

            Assert(manager->numJobsAddedToProcessor >= 1);
            Assert(manager->isWaitable);
            Assert(GetCurrentJobOfManager(manager));

        }

        //Wait for all the on going jobs to complete.
        criticalSection.Enter();
        while (true)
        {
            Job *job = GetCurrentJobOfManager(manager);
            if (!job)
            {
                break;
            }

            WaitableJobManager * const waitableManager = static_cast<WaitableJobManager *>(manager);
            Assert(!waitableManager->jobBeingWaitedUpon);

            waitableManager->jobBeingWaitedUpon = job;
            waitableManager->jobBeingWaitedUponProcessed.Reset();

            threadDataProcessingCurrentJob = GetThreadDataFromCurrentJob(waitableManager->jobBeingWaitedUpon);
            criticalSection.Leave();

            WaitWithThread(threadDataProcessingCurrentJob, waitableManager->jobBeingWaitedUponProcessed);

            criticalSection.Enter();
            waitableManager->jobBeingWaitedUpon = 0;
        }
        criticalSection.Leave();
    }

    void BackgroundJobProcessor::AddJob(Job *const job, const bool prioritize)
    {
        // This function is called from inside the lock

        Assert(job);
        Assert(managers.Contains(job->Manager()));
        Assert(!IsClosed());

        if(numJobs + 1 == 0)
            Js::Throw::OutOfMemory(); // Overflow: job counts we use are int32's.
        ++numJobs;

        __super::AddJob(job, prioritize);
        IndicateNewJob();
    }

    bool BackgroundJobProcessor::RemoveJob(Job *const job)
    {
        // This function is called from inside the lock

        Assert(job);
        Assert(managers.Contains(job->Manager()));
        Assert(!IsClosed());

        if (IsBeingProcessed(job))
        {
            return false;
        }
        return __super::RemoveJob(job);
    }

    bool BackgroundJobProcessor::Process(Job *const job, ParallelThreadData *threadData)
    {
        try
        {
            AUTO_HANDLED_EXCEPTION_TYPE(static_cast<ExceptionType>(ExceptionType_OutOfMemory | ExceptionType_StackOverflow));
            return job->Manager()->Process(job, threadData);
        }
        catch(Js::OutOfMemoryException)
        {
            // Treat OOM to be a non-terminal failure
#if ENABLE_DEBUG_CONFIG_OPTIONS
            job->failureReason = Job::FailureReason::OOM;
#endif
        }
        catch(Js::StackOverflowException)
        {
            // Treat stack overflow to be a non-terminal failure
#if ENABLE_DEBUG_CONFIG_OPTIONS
            job->failureReason = Job::FailureReason::StackOverflow;
#endif
        }
        catch(Js::OperationAbortedException)
        {
            // This can happen for any reason a job needs to be aborted while executing, like for instance, if the script
            // context is closed while the job is being processed in the background
#if ENABLE_DEBUG_CONFIG_OPTIONS
            job->failureReason = Job::FailureReason::Aborted;
#endif
        }

        // Since the background job processor processes jobs on a background thread, out-of-memory and stack overflow need to be
        // caught here. Script would not be active in the background thread, so (at the time of this writing) a
        // JavascriptException would never be thrown and instead the corresponding exceptions caught above would be thrown.

        // Any of the above exceptions will result in the job failing. The return value of this function will cause the job
        // manager to get a JobProcessed call with succeeded = false, so that it can handle the failure appropriately.
        return false;
    }

    void BackgroundJobProcessor::Run(ParallelThreadData* threadData)
    {
        EDGE_ETW_INTERNAL(EventWriteJSCRIPT_NATIVECODEGEN_START(this, 0));

        ArenaAllocator threadArena(_u("ThreadArena"), threadData->GetPageAllocator(), Js::Throw::OutOfMemory);
        threadData->threadArena = &threadArena;

        {
            // Make sure we take decommit action before the threadArena is torn down, in case the
            // thread context goes away and the loop exits.
            struct AutoDecommit
            {
                AutoDecommit(JobProcessor *proc, ParallelThreadData *data) : processor(proc), threadData(data) {}
                ~AutoDecommit()
                {
                    processor->ForEachManager([this](JobManager *manager){
                        manager->OnDecommit(this->threadData);
                    });
                }
                ParallelThreadData *threadData;
                JobProcessor *processor;
            } autoDecommit(this, threadData);

            criticalSection.Enter();
            while (!IsClosed() || (jobs.Head() && jobs.Head()->IsCritical()))
            {
                Job *job = jobs.UnlinkFromBeginning();

                if(!job)
                {
                    // No jobs in queue, wait for one

                    Assert(!IsClosed());
                    Assert(!threadData->isWaitingForJobs);
                    threadData->isWaitingForJobs = true;
                    criticalSection.Leave();
                    EDGE_ETW_INTERNAL(EventWriteJSCRIPT_NATIVECODEGEN_STOP(this, 0));

                    if (threadService->HasCallback())
                    {
                        // We have a thread service, so simply return the thread back now.
                        // When new jobs are submitted, we will be called to process again.
                        return;
                    }

                    WaitForJobReadyOrShutdown(threadData);

                    EDGE_ETW_INTERNAL(EventWriteJSCRIPT_NATIVECODEGEN_START(this, 0));
                    criticalSection.Enter();
                    threadData->isWaitingForJobs = false;
                    continue;
                }
                else
                {
                    // Job found. Proceed with Processing
                    Assert(numJobs != 0);
                    --numJobs;
                    threadData->currentJob = job;

                    JobManager *const manager = job->Manager();
                    manager->JobProcessing(job);

                    criticalSection.Leave();

                    const bool succeeded = Process(job, threadData);

                    criticalSection.Enter();
                    threadData->currentJob = 0;
                    
                    JobProcessed(manager, job, succeeded); // the job may be deleted during this and should not be used afterwards
                    Assert(manager->numJobsAddedToProcessor != 0);
                    --manager->numJobsAddedToProcessor;
                    if (manager->isWaitable)
                    {
                        WaitableJobManager *const waitableManager = static_cast<WaitableJobManager *>(manager);
                        Assert(!(waitableManager->jobBeingWaitedUpon && waitableManager->isWaitingForQueuedJobs));
                        if (waitableManager->jobBeingWaitedUpon == job)
                        {
                            waitableManager->jobBeingWaitedUpon = 0;
                            waitableManager->jobBeingWaitedUponProcessed.Set();
                        }
                        else if (waitableManager->isWaitingForQueuedJobs && manager->numJobsAddedToProcessor == 0)
                        {
                            waitableManager->isWaitingForQueuedJobs = false;
                            waitableManager->queuedJobsProcessed.Set();
                        }
                    }
                    if (manager->numJobsAddedToProcessor == 0)
                    {
                        LastJobProcessed(manager); // the manager may be deleted during this and should not be used afterwards
                    }
                }
            }
            criticalSection.Leave();

#if PDATA_ENABLED && defined(_WIN32)
            // flush the function tables in background thread after closed and before shutting down thread
            DelayDeletingFunctionTable::Clear();
#endif

            EDGE_ETW_INTERNAL(EventWriteJSCRIPT_NATIVECODEGEN_STOP(this, 0));
        }
    }

    void BackgroundJobProcessor::Close()
    {
        // The contract for Close is that from the time it's called, job managers and jobs may no longer be added to the job
        // processor. If there is potential background work that needs to be done after Close, it must be done directly on the
        // foreground thread.

        if(IsClosed())
            return;

        bool waitForThread = true;
        uint threadsWaitingForJobs = 0;

        {
            AutoCriticalSection lock(&criticalSection);
            if(IsClosed())
                return;

            Job *nextJob = jobs.Head();
            while(nextJob)
            {
                Job *const job = nextJob;
                nextJob = job->Next();
                if(job->IsCritical())
                {
                    // Critical jobs need to be left in the queue. After this instance is flagged as closed, the background
                    // thread will continue processing critical jobs, for which this function will wait before returning.
                    continue;
                }

                jobs.Unlink(job);
                Assert(numJobs != 0);
                --numJobs;
                JobManager *const manager = job->Manager();
                JobProcessed(manager, job, false); // the job may be deleted during this and should not be used afterwards
                Assert(manager->numJobsAddedToProcessor != 0);
                --manager->numJobsAddedToProcessor;
                if(manager->isWaitable)
                {
                    WaitableJobManager *const waitableManager = static_cast<WaitableJobManager *>(manager);
                    if(waitableManager->jobBeingWaitedUpon == job)
                    {
                        waitableManager->jobBeingWaitedUponProcessed.Set();
                        waitableManager->jobBeingWaitedUpon = 0;
                    }
                }
                if(manager->numJobsAddedToProcessor == 0)
                {
                    Assert(!GetCurrentJobOfManager(manager));
                    LastJobProcessed(manager); // the manager may be deleted during this and should not be used afterwards
                }
            }

            // Managers will remove themselves, so not removing managers here

            JobProcessor::Close();

            if (threadService->HasCallback())
            {
                Assert(this->threadCount == 1);
                // If there are no outstanding jobs, then we don't currently have a thread, so there's no reason to wait for it.
                waitForThread = !this->parallelThreadData[0]->isWaitingForJobs;
            }
            else
            {
                threadsWaitingForJobs = NumberOfThreadsWaitingForJobs ();
            }
        }

        if (threadsWaitingForJobs)
        {
            //There is no reset for this. It will be signaled until all the threads get out of their hibernation.
            wakeAllBackgroundThreads.Set();
        }

        // We cannot wait for the background thread to terminate because this function may be called from DLL_THREAD_DETACH, and
        // waiting for the background thread would then deadlock because the background thread would also be blocked from
        // detaching. Instead, we just wait for this event to be signaled, which indicates that the thread will promptly end
        // naturally. The caller should wait as necessary for the thread to terminate.
        if (waitForThread)
        {
            WaitWithAllThreadsForThreadStartedOrClosingEvent();
        }

#if DBG
        if (!threadService->HasCallback())
        {
            AutoCriticalSection lock(&criticalSection);
            Assert(!NumberOfThreadsWaitingForJobs());

            this->IterateBackgroundThreads([](ParallelThreadData *threadData)
                {
                    threadData->backgroundPageAllocator.ClearConcurrentThreadId();
                    return false;
                }
            );
        }
#endif

        if (threadService->HasCallback())
        {
            Assert(this->threadCount == 1 && (this->parallelThreadData[0]->threadHandle == 0));
            return;
        }
        else
        {
            // Close all the handles
            this->IterateBackgroundThreads([&](ParallelThreadData *threadData)
            {
                CloseHandle(threadData->threadHandle);
                threadData->threadHandle = 0;
                return false;
            });
        }
    }

    unsigned int WINAPI BackgroundJobProcessor::StaticThreadProc(void *lpParam)
    {
        Assert(lpParam);

#ifdef TARGET_64
#ifdef RECYCLER_WRITE_BARRIER
        Memory::RecyclerWriteBarrierManager::OnThreadInit();
#endif
#endif

#if !defined(_UCRT)
        HMODULE dllHandle = NULL;
        if (!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)&BackgroundJobProcessor::StaticThreadProc, &dllHandle))
        {
            dllHandle = NULL;
        }
#endif

        ParallelThreadData * threadData = static_cast<ParallelThreadData *>(lpParam);
        BackgroundJobProcessor *const processor = threadData->processor;

        // Indicate to the constructor that the thread has fully started.
        threadData->threadStartedOrClosing.Set();

#if DBG
        threadData->backgroundPageAllocator.SetConcurrentThreadId(GetCurrentThreadId());
#endif

#ifdef DISABLE_SEH
        processor->Run(threadData);
#else
        __try
        {
            processor->Run(threadData);
        }
        __except(ExceptFilter(GetExceptionInformation()))
        {
            Assert(false);
        }
#endif

        // Indicate to Close that the thread is about to exit. This has to be done before CoUninitialize because CoUninitialize
        // may require the loader lock and if Close was called while holding the loader lock during DLL_THREAD_DETACH, it could
        // end up waiting forever, causing a deadlock.
        threadData->threadStartedOrClosing.Set();
#if !defined(_UCRT)
        if (dllHandle)
        {
            FreeLibraryAndExitThread(dllHandle, 0);
        }
        else
#endif
        {
            return 0;
        }
    }

#ifndef DISABLE_SEH
    int BackgroundJobProcessor::ExceptFilter(LPEXCEPTION_POINTERS pEP)
    {
#if DBG && defined(_WIN32)
        // Assert exception code
        if (pEP->ExceptionRecord->ExceptionCode == STATUS_ASSERTION_FAILURE)
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
#endif

#ifdef GENERATE_DUMP
        if (Js::Configuration::Global.flags.IsEnabled(Js::DumpOnCrashFlag))
        {
            Js::Throw::GenerateDump(pEP, Js::Configuration::Global.flags.DumpOnCrash);
        }
#endif

#if DBG && _M_IX86
        int callerEBP = *((int*)pEP->ContextRecord->Ebp);

        Output::Print(_u("BackgroundJobProcessor: Uncaught exception: EIP: 0x%X  ExceptionCode: 0x%X  EBP: 0x%X  ReturnAddress: 0x%X  ReturnAddress2: 0x%X\n"),
            pEP->ExceptionRecord->ExceptionAddress, pEP->ExceptionRecord->ExceptionCode, pEP->ContextRecord->Eip,
            pEP->ContextRecord->Ebp, *((int*)pEP->ContextRecord->Ebp + 1), *((int*) callerEBP + 1));
#endif
        Output::Flush();
        return EXCEPTION_CONTINUE_SEARCH;
    }
#endif

    void BackgroundJobProcessor::ThreadServiceCallback(void * callbackData)
    {
        BackgroundJobProcessor * jobProcessor = (BackgroundJobProcessor *)callbackData;

        Assert(jobProcessor->threadCount == 1);
#if DBG
        jobProcessor->parallelThreadData[0]->backgroundPageAllocator.SetConcurrentThreadId(GetCurrentThreadId());
#endif

        jobProcessor->Run(jobProcessor->parallelThreadData[0]);

#if DBG
        jobProcessor->parallelThreadData[0]->backgroundPageAllocator.ClearConcurrentThreadId();
#endif

        // Set the thread event, in case we are waiting for it on shutdown.
        jobProcessor->parallelThreadData[0]->threadStartedOrClosing.Set();
    }

    bool BackgroundJobProcessor::AreAllThreadsWaitingForJobs()
    {
        Assert(criticalSection.IsLocked());

        bool isAnyThreadNotWaitingForJobs = false;

        this->IterateBackgroundThreads([&](ParallelThreadData *parallelThreadData)
        {
            if (parallelThreadData->isWaitingForJobs)
            {
                return false;
            }
            // At least one thread was not waiting for jobs.
            isAnyThreadNotWaitingForJobs = true;
            return true;
        });

        return !isAnyThreadNotWaitingForJobs;
    }

    uint BackgroundJobProcessor::NumberOfThreadsWaitingForJobs ()
    {
        Assert(criticalSection.IsLocked());

        uint countOfThreadsWaitingForJobs = 0;

        this->IterateBackgroundThreads([&](ParallelThreadData *parallelThreadData)
        {
            if (parallelThreadData->isWaitingForJobs)
            {
                // At least one thread is waiting for jobs.
                countOfThreadsWaitingForJobs++;
            }
            return false;
        });

        return countOfThreadsWaitingForJobs;
    }

    bool BackgroundJobProcessor::IsBeingProcessed(Job* job)
    {
        Assert(criticalSection.IsLocked());

        bool isBeingProcessed = false;

        this->IterateBackgroundThreads([&](ParallelThreadData *parallelThreadData)
        {
            if (parallelThreadData->currentJob == job)
            {
                isBeingProcessed = true;
                return true;
            }
            return false;
        });

        return isBeingProcessed;
    }

    void BackgroundJobProcessor::AssociatePageAllocator(PageAllocator* const pageAllocator)
    {
#if DBG
        pageAllocator->SetConcurrentThreadId(::GetCurrentThreadId());
#endif
    }

    void BackgroundJobProcessor::DissociatePageAllocator(PageAllocator* const pageAllocator)
    {
        // This function is called from the foreground thread
#if DBG
        // Assert that the dissociation is happening in the same thread that created the background job processor
        Assert(GetCurrentThreadContextId() == this->threadId);
        pageAllocator->ClearConcurrentThreadId();
#endif
    }

#if PDATA_ENABLED && defined(_WIN32)
    void BackgroundJobProcessor::StartExtraWork()
    {
        InterlockedIncrement(&hasExtraWork);

        // Signal the background thread to wake up and process the extra work.
        jobReady.Set();
    }
    void BackgroundJobProcessor::EndExtraWork()
    {
        LONG newValue = InterlockedDecrement(&hasExtraWork);
        Assert(newValue >= 0);
    }
#endif

#if DBG_DUMP
    //Just for debugging purpose
    char16 const * const  BackgroundJobProcessor::DebugThreadNames[16] = {
        _u("BackgroundJobProcessor thread 1"),
        _u("BackgroundJobProcessor thread 2"),
        _u("BackgroundJobProcessor thread 3"),
        _u("BackgroundJobProcessor thread 4"),
        _u("BackgroundJobProcessor thread 5"),
        _u("BackgroundJobProcessor thread 6"),
        _u("BackgroundJobProcessor thread 7"),
        _u("BackgroundJobProcessor thread 8"),
        _u("BackgroundJobProcessor thread 9"),
        _u("BackgroundJobProcessor thread 10"),
        _u("BackgroundJobProcessor thread 11"),
        _u("BackgroundJobProcessor thread 12"),
        _u("BackgroundJobProcessor thread 13"),
        _u("BackgroundJobProcessor thread 14"),
        _u("BackgroundJobProcessor thread 15"),
        _u("BackgroundJobProcessor thread 16") };
#endif
#endif // ENABLE_BACKGROUND_JOB_PROCESSOR
}
