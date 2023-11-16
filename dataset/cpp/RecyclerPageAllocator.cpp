//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "CommonMemoryPch.h"

RecyclerPageAllocator::RecyclerPageAllocator(HeapInfo * heapInfo, AllocationPolicyManager * policyManager,
    Js::ConfigFlagsTable& flagTable, uint maxFreePageCount, uint maxAllocPageCount, bool enableWriteBarrier)
    : IdleDecommitPageAllocator(policyManager,
        PageAllocatorType_Recycler,
        flagTable,
        0, maxFreePageCount,
        true,
#if ENABLE_BACKGROUND_PAGE_ZEROING
        &zeroPageQueue,
#endif
        maxAllocPageCount,
        enableWriteBarrier
        )
{
    this->heapInfo = heapInfo;
}

bool RecyclerPageAllocator::IsMemProtectMode()
{
    return heapInfo->GetRecycler()->IsMemProtectMode();
}

#if ENABLE_CONCURRENT_GC
#ifdef RECYCLER_WRITE_WATCH
void
RecyclerPageAllocator::EnableWriteWatch()
{
    Assert(segments.Empty());
    Assert(fullSegments.Empty());
    Assert(emptySegments.Empty());
    Assert(decommitSegments.Empty());
    Assert(largeSegments.Empty());

    allocFlags = MEM_WRITE_WATCH;
}

bool
RecyclerPageAllocator::ResetWriteWatch()
{
    if (!IsWriteWatchEnabled())
    {
        return false;
    }

    GCETW(GC_RESETWRITEWATCH_START, (this));

    SuspendIdleDecommit();

    bool success = true;
    // Only reset write watch on allocated pages
    if (!ResetWriteWatch(&segments) ||
        !ResetWriteWatch(&decommitSegments) ||
        !ResetAllWriteWatch(&fullSegments) ||
        !ResetAllWriteWatch(&largeSegments))
    {
        allocFlags = 0;
        success = false;
    }

    ResumeIdleDecommit();

    GCETW(GC_RESETWRITEWATCH_STOP, (this));

    return success;
}

bool
RecyclerPageAllocator::ResetWriteWatch(DListBase<PageSegment> * segmentList)
{
    DListBase<PageSegment>::Iterator i(segmentList);
    while (i.Next())
    {
        PageSegment& segment = i.Data();
        size_t pageCount = segment.GetAvailablePageCount();
        Assert(pageCount <= MAXUINT32);
        PageSegment::PageBitVector unallocPages = segment.GetUnAllocatedPages();
        for (uint index = 0u; index < pageCount; index++)
        {
            if (unallocPages.Test(index))
            {
                continue;
            }
            char * address = segment.GetAddress() + index * AutoSystemInfo::PageSize;
            if (::ResetWriteWatch(address, AutoSystemInfo::PageSize) != 0)

            {
#if DBG_DUMP
                Output::Print(_u("ResetWriteWatch failed for %p\n"), address);
                Output::Flush();
#endif
                // shouldn't happen
                Assert(false);
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool
RecyclerPageAllocator::ResetAllWriteWatch(DListBase<T> * segmentList)
{
    typename DListBase<T>::Iterator i(segmentList);
    while (i.Next())
    {
        T& segment = i.Data();
        if (::ResetWriteWatch(segment.GetAddress(),  segment.GetPageCount() * AutoSystemInfo::PageSize ) != 0)
        {
#if DBG_DUMP
            Output::Print(_u("ResetWriteWatch failed for %p\n"), segment.GetAddress());
            Output::Flush();
#endif
            // shouldn't happen
            Assert(false);
            return false;
        }
    }
    return true;
}
#endif

#ifdef RECYCLER_WRITE_WATCH
#if DBG
size_t
RecyclerPageAllocator::GetWriteWatchPageCount()
{
    if (allocFlags != MEM_WRITE_WATCH)
    {
        return 0;
    }

    SuspendIdleDecommit();

    // Only reset write watch on allocated pages
    size_t count = GetWriteWatchPageCount(&segments)
        + GetWriteWatchPageCount(&decommitSegments)
        + GetAllWriteWatchPageCount(&fullSegments)
        + GetAllWriteWatchPageCount(&largeSegments);

    ResumeIdleDecommit();

    return count;
}


size_t
RecyclerPageAllocator::GetWriteWatchPageCount(DListBase<PageSegment> * segmentList)
{
    size_t totalCount = 0;
    DListBase<PageSegment>::Iterator i(segmentList);
    while (i.Next())
    {
        PageSegment& segment = i.Data();
        size_t pageCount = segment.GetAvailablePageCount();
        Assert(pageCount <= MAXUINT32);
        PageSegment::PageBitVector unallocPages = segment.GetUnAllocatedPages();
        for (uint index = 0u; index < pageCount; index++)
        {
            if (unallocPages.Test(index))
            {
                continue;
            }
            char * address = segment.GetAddress() + index * AutoSystemInfo::PageSize;
            void * written;
            ULONG_PTR count = 0;
            DWORD pageSize = AutoSystemInfo::PageSize;
            if (::GetWriteWatch(0, address, AutoSystemInfo::PageSize, &written, &count, &pageSize) == 0)
            {
#if DBG_DUMP
                Output::Print(_u("GetWriteWatch failed for %p\n"), segment.GetAddress());
                Output::Flush();
#endif
                // shouldn't happen
                Assert(false);
            }
            else
            {
                Assert(count <= 1);
                Assert(pageSize == AutoSystemInfo::PageSize);
                Assert(count == 0 || written == address);
                totalCount += count;
            }
        }
    }
    return totalCount;
}

template <typename T>
size_t
RecyclerPageAllocator::GetAllWriteWatchPageCount(DListBase<T> * segmentList)
{
    size_t totalCount = 0;
    _TYPENAME DListBase<T>::Iterator it(segmentList);
    while (it.Next())
    {
        T& segment = it.Data();
        for (uint i = 0; i < segment.GetPageCount(); i++)
        {
            void * address = segment.GetAddress() + i * AutoSystemInfo::PageSize;
            void * written;
            ULONG_PTR count = 0;
            DWORD pageSize = AutoSystemInfo::PageSize;
            if (::GetWriteWatch(0, address, AutoSystemInfo::PageSize, &written, &count, &pageSize) == 0)
            {
#if DBG_DUMP
                Output::Print(_u("GetWriteWatch failed for %p\n"), segment.GetAddress());
                Output::Flush();
#endif
                // shouldn't happen
                Assert(false);
            }
            else
            {
                Assert(count <= 1);
                Assert(pageSize == AutoSystemInfo::PageSize);
                Assert(count == 0 || written == address);
                totalCount += count;
            }
        }
    }
    return totalCount;
}
#endif
#endif
#endif
