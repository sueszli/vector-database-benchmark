//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeBasePch.h"
#include "Base/ThreadServiceWrapperBase.h"

ThreadServiceWrapperBase::ThreadServiceWrapperBase() :
    threadContext(nullptr),
    needIdleCollect(false),
    inIdleCollect(false),
    hasScheduledIdleCollect(false),
    shouldScheduleIdleCollectOnExitIdle(false),
    forceIdleCollectOnce(false)
{
}

bool ThreadServiceWrapperBase::Initialize(ThreadContext *newThreadContext)
{
    if (newThreadContext == nullptr)
    {
        return false;
    }
    threadContext = newThreadContext;
    threadContext->SetThreadServiceWrapper(this);
    return true;
}

void ThreadServiceWrapperBase::Shutdown()
{
    if (hasScheduledIdleCollect)
    {
#if DBG
        // Fake the inIdleCollect to get pass asserts in FinishIdleCollect
        inIdleCollect = true;
#endif
        FinishIdleCollect(FinishReason::FinishReasonNormal);
    }
}

bool ThreadServiceWrapperBase::ScheduleIdleCollect(uint ticks, bool scheduleAsTask)
{
    Assert(!threadContext->IsInScript());

    // We should schedule have called this in one of two cases:
    //  1) Either needIdleCollect is true- in which case, we should schedule one
    //  2) Or ScheduleNextCollectionOnExit was called when needIdleCollect was true, but we didn't schedule
    //      one because we were at the time in one. Later, as we unwound, we might have set needIdleCollect to false
    //      but because we had noted that we needed to schedule a collect, we would end up coming into this function
    //      so allow for that
    Assert(needIdleCollect || shouldScheduleIdleCollectOnExitIdle || threadContext->GetRecycler()->CollectionInProgress());

    if (!CanScheduleIdleCollect())
    {
        return false;
    }

    if (hasScheduledIdleCollect)
    {
        return true;
    }

    if (OnScheduleIdleCollect(ticks, scheduleAsTask))
    {
        JS_ETW(EventWriteJSCRIPT_GC_IDLE_START(this));
        IDLE_COLLECT_VERBOSE_TRACE(_u("ScheduledIdleCollect- Set hasScheduledIdleCollect\n"));

        hasScheduledIdleCollect = true;
        return true;
    }
    else
    {
        IDLE_COLLECT_TRACE(_u("Idle timer setup failed\n"));
        FinishIdleCollect(FinishReason::FinishReasonIdleTimerSetupFailed);
        return false;
    }
}

bool ThreadServiceWrapperBase::IdleCollect()
{
    // Tracking service does not AddRef/Release the thread service and only keeps a function pointer and context parameter (this pointer)
    // to execute the IdleCollect callback. It is possible that the tracking service gets destroyed as part of the collection
    // during this IdleCollect. If that happens then we need to make sure ThreadService (which may be owned by the tracking service)
    // is kept alive until this callback completes. Any pending timer is killed in the thread service destructor so we should not get
    // any new callbacks after the thread service is destroyed.
    AutoAddRefReleaseThreadService autoThreadServiceKeepAlive(this);

    Assert(hasScheduledIdleCollect);
    IDLE_COLLECT_VERBOSE_TRACE(_u("IdleCollect- reset hasScheduledIdleCollect\n"));
    hasScheduledIdleCollect = false;

    // Don't do anything and kill the timer if we are called recursively or if we are in script
    if (inIdleCollect || threadContext->IsInScript())
    {
        FinishIdleCollect(FinishReason::FinishReasonNormal);
        return hasScheduledIdleCollect;
    }

    // If during idle collect we determine that we need to schedule another
    // idle collect, this gets flipped to true
    shouldScheduleIdleCollectOnExitIdle = false;

    AutoBooleanToggle autoInIdleCollect(&inIdleCollect);
    Recycler* recycler = threadContext->GetRecycler();
#if ENABLE_CONCURRENT_GC
    // Finish concurrent on timer heart beat if needed
    // We wouldn't try to finish if we need to schedule
    // an idle task to finish the collection
    if (this->ShouldFinishConcurrentCollectOnIdleCallback() && recycler->FinishConcurrent<FinishConcurrentOnIdle>())
    {
        IDLE_COLLECT_TRACE(_u("Idle callback: finish concurrent\n"));
        JS_ETW(EventWriteJSCRIPT_GC_IDLE_CALLBACK_FINISH(this));
    }
#endif

    while (true)
    {
        // If a GC is still happening, just wait for the next heart beat
        if (recycler->CollectionInProgress())
        {
            ScheduleIdleCollect(IdleTicks, true /* schedule as task */);
            break;
        }

        // If there no more need of idle collect, then cancel the timer
        if (!needIdleCollect)
        {
            FinishIdleCollect(FinishReason::FinishReasonNormal);
            break;
        }

        int timeDiff = tickCountNextIdleCollection - GetTickCount();

        // See if we pass the time for the next scheduled Idle GC
        if (timeDiff > 0)
        {
            // Not time yet, wait for the next heart beat
            ScheduleIdleCollect(IdleTicks, false /* not schedule as task */);

            IDLE_COLLECT_TRACE(_u("Idle callback: nop until next collection: %d\n"), timeDiff);
            break;
        }

        // activate an idle collection
        IDLE_COLLECT_TRACE(_u("Idle callback: collection: %d\n"), timeDiff);
        JS_ETW(EventWriteJSCRIPT_GC_IDLE_CALLBACK_NEWCOLLECT(this));

        needIdleCollect = false;
        recycler->CollectNow<CollectOnScriptIdle>();
    }

    if (shouldScheduleIdleCollectOnExitIdle)
    {
        ScheduleIdleCollect(IdleTicks, false /* not schedule as task */);
    }

    return hasScheduledIdleCollect;
}

void ThreadServiceWrapperBase::FinishIdleCollect(ThreadServiceWrapperBase::FinishReason reason)
{
    Assert(reason == FinishReason::FinishReasonIdleTimerSetupFailed ||
        reason == FinishReason::FinishReasonTaskComplete ||
        inIdleCollect || threadContext->IsInScript() || !threadContext->GetRecycler()->CollectionInProgress());

    IDLE_COLLECT_VERBOSE_TRACE(_u("FinishIdleCollect- Reset hasScheduledIdleCollect\n"));
    hasScheduledIdleCollect = false;
    needIdleCollect = false;

    OnFinishIdleCollect();

    IDLE_COLLECT_TRACE(_u("Idle timer finished\n"));

    if (reason == FinishReason::FinishReasonTaskComplete 
        && threadContext->GetRecycler()->CollectionInProgress() 
        && !threadContext->IsInScript())
    {
        // schedule another timer to check the progress
        IDLE_COLLECT_TRACE(_u("FinishIdleCollect- collection is still in progress, schedule another timer to check the status\n"));
        // schedule a shorter timer here to check the progress since the idle GC has already last more than one second and is likely to close to finish
        // TODO: measure and adjust the timeout bellow
        ScheduleIdleCollect(RecyclerHeuristic::TickCountIdleCollectRepeatTimer, false);
    }
    else
    {
        JS_ETW(EventWriteJSCRIPT_GC_IDLE_FINISHED(this));
    }
}

bool ThreadServiceWrapperBase::ScheduleNextCollectOnExit()
{
    Assert(!threadContext->IsInScript());
    Assert(!needIdleCollect || hasScheduledIdleCollect);

    Recycler* recycler = threadContext->GetRecycler();
#if ENABLE_CONCURRENT_GC
    recycler->FinishConcurrent<FinishConcurrentOnExitScript>();
#endif

#ifdef RECYCLER_TRACE
    bool oldNeedIdleCollect = needIdleCollect;

    if (forceIdleCollectOnce)
    {
        IDLE_COLLECT_VERBOSE_TRACE(_u("Need to force one idle collection\n"));
    }
#endif

    needIdleCollect = forceIdleCollectOnce || recycler->ShouldIdleCollectOnExit();

    if (needIdleCollect)
    {
        // Set up when we will do the idle decommit
        tickCountNextIdleCollection = GetTickCount() + IdleTicks;

        IDLE_COLLECT_VERBOSE_TRACE(_u("Idle on exit collect %s: %d\n"), (oldNeedIdleCollect ? _u("rescheduled") : _u("scheduled")),
            tickCountNextIdleCollection - GetTickCount());

        JS_ETW(EventWriteJSCRIPT_GC_IDLE_SCHEDULED(this));
    }
    else
    {
        IDLE_COLLECT_VERBOSE_TRACE(_u("Idle on exit collect %s\n"), oldNeedIdleCollect ? _u("cancelled") : _u("not scheduled"));
        if (!recycler->CollectionInProgress())
        {
            // We collected and finished, no need to ensure the idle collect call back.
            return true;
        }

        IDLE_COLLECT_VERBOSE_TRACE(_u("Idle on exit collect %s\n"), hasScheduledIdleCollect || oldNeedIdleCollect ? _u("reschedule finish") : _u("schedule finish"));
    }

    // Don't schedule the call back if we are already in idle call back, as we don't do anything on recursive call anyways
    // IdleCollect will schedule one if necessary
    if (inIdleCollect)
    {
        shouldScheduleIdleCollectOnExitIdle = true;
        return true;
    }
    else
    {
        return ScheduleIdleCollect(IdleTicks, false /* not schedule as task */);
    }
}

void ThreadServiceWrapperBase::ClearForceOneIdleCollection()
{
    IDLE_COLLECT_VERBOSE_TRACE(_u("Clearing force idle collect flag\n"));

    this->forceIdleCollectOnce = false;
}

void ThreadServiceWrapperBase::SetForceOneIdleCollection()
{
    IDLE_COLLECT_VERBOSE_TRACE(_u("Setting force idle collect flag\n"));

    this->forceIdleCollectOnce = true;
}

void ThreadServiceWrapperBase::ScheduleFinishConcurrent()
{
    Assert(!threadContext->IsInScript());
    Assert(threadContext->GetRecycler()->CollectionInProgress());

    if (!this->inIdleCollect)
    {
        IDLE_COLLECT_VERBOSE_TRACE(_u("Idle collect %s\n"), needIdleCollect ? _u("reschedule finish") : _u("scheduled finish"));
        this->needIdleCollect = false;
        ScheduleIdleCollect(IdleFinishTicks, true /* schedule as task */);
    }
}
