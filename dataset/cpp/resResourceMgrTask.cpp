#include "KingSystem/Resource/resResourceMgrTask.h"
#include <framework/seadHeapPolicies.h>
#include <framework/seadTaskID.h>
#include <resource/seadResourceMgr.h>
#include <resource/seadSZSDecompressor.h>
#include <thread/seadThreadUtil.h>
#include "KingSystem/Framework/frmWorkerSupportThreadMgr.h"
#include "KingSystem/Resource/resCache.h"
#include "KingSystem/Resource/resCompactedHeap.h"
#include "KingSystem/Resource/resEntryFactory.h"
#include "KingSystem/Resource/resMemoryTask.h"
#include "KingSystem/Resource/resSystem.h"
#include "KingSystem/Resource/resTextureHandleList.h"
#include "KingSystem/Resource/resTextureHandleMgr.h"
#include "KingSystem/System/OverlayArenaSystem.h"
#include "KingSystem/Utils/SafeDelete.h"
#include "KingSystem/Utils/Thread/GameTaskThread.h"
#include "KingSystem/Utils/Thread/TaskMgr.h"
#include "KingSystem/Utils/Thread/TaskQueueBase.h"
#include "KingSystem/Utils/Thread/TaskQueueLock.h"
#include "KingSystem/Utils/Thread/TaskThread.h"

namespace ksys::res {

namespace {
class ClearCachesTaskData : public util::TaskData {
    SEAD_RTTI_OVERRIDE(ClearCachesTaskData, util::TaskData)
public:
    virtual ~ClearCachesTaskData() = default;

    bool _8;
    s32 _c;
    sead::SafeString mStr;
};
KSYS_CHECK_SIZE_NX150(ClearCachesTaskData, 0x20);
}  // namespace

void ResourceMgrTask::setInstance(ResourceMgrTask* task) {
    if (!sInstance) {
        sInstance = task;
        task->mInstancePtrClearer.mClearOnDestruction = true;
    }
}

ResourceMgrTask::ResourceMgrTask(const sead::TaskConstructArg& arg)
    : sead::CalculateTask(arg, "res::ResourceMgrTask"),
      mSzsDecompressorCS(arg.heap_array->getPrimaryHeap()),
      mCounter(arg.heap_array->getPrimaryHeap()), mTask(arg.heap_array->getPrimaryHeap()) {
    mArenas.initOffset(OverlayArena::getListNodeOffset());
    mUnits.initOffset(ResourceUnit::getResMgrUnitListNodeOffset());
    mSomeList.initOffset(0x188);  // TODO: replace this with a "get offset" call
    mSystemCalcFn.bind(this, &ResourceMgrTask::callSystemCalc_);
    mFileDevicePrefixes.initOffset(FileDevicePrefix::getListNodeOffset());
}

ResourceMgrTask::~ResourceMgrTask() {
    util::safeDelete(mTexHandleList);
    util::safeDelete(mTexHandleMgr);
    util::safeDeleteThread(mCompactionThread);

    if (mCompactedHeapMip0) {
        mCompactedHeapMip0->destroy();
        util::safeDeleteArray(mCompactedHeapMip0Buffer);
    }

    if (mCompactedHeapMain) {
        mCompactedHeapMain->destroy();
        util::safeDeleteArray(mCompactedHeapMainBuffer);
        mCompactedHeapMainSeadHeap->destroy();
    }

    util::safeDeleteArray(mCompactedHeapMainBuffer2);
    util::safeDelete(mOffsetReadBuf);
    mExtensions2.freeBuffer();
    mExtensions1.freeBuffer();

    util::safeDeleteThread(mMovableMemoryThread);
    util::safeDeleteThread(mResourceMemoryThread);
    util::safeDeleteThread(mResourceControlThread);
    util::safeDeleteThread(mResourceLoadingThread);
    util::safeDelete(mTask3);
    util::safeDelete(mTask2);
    util::safeDelete(mTask1);
    util::safeDelete(mControlTask);

    if (mResourceMemoryTaskMgr) {
        mResourceMemoryTaskMgr->finalize();
        util::safeDelete(mResourceMemoryTaskMgr);
    }

    if (mResourceControlTaskMgr) {
        mResourceControlTaskMgr->finalize();
        util::safeDelete(mResourceControlTaskMgr);
    }

    util::safeDelete(mEntryFactoryBase);

    mResSystemHeap->destroy();
}

void ResourceMgrTask::insertOverlayArena(OverlayArena* arena) {
    auto lock = sead::makeScopedLock(mArenasCS);
    if (!mArenas.isNodeLinked(arena)) {
        mArenas.pushBack(arena);
        stubbedLogFunction();
    }
}

util::TaskThread* ResourceMgrTask::makeResourceLoadingThread(sead::Heap* heap,
                                                             bool use_game_task_thread) {
    if (use_game_task_thread) {
        return new (heap) util::GameTaskThread(
            "Resource Loading", heap, sead::ThreadUtil::ConvertPrioritySeadToPlatform(19),
            sead::MessageQueue::BlockType::Blocking, 0x7fffffff, 0xfa000, 32);
    }

    return new (heap) util::TaskThread(
        "Resource Loading", heap, sead::ThreadUtil::ConvertPrioritySeadToPlatform(19),
        sead::MessageQueue::BlockType::Blocking, 0x7fffffff, 0xa000, 32);
}

void ResourceMgrTask::clearAllCaches(OverlayArena* arena) {
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_5));
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_4));
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_3));

    MemoryTaskRequest req;
    req.mLaneId = u8(LaneId::_9);
    req.mHasHandle = true;
    req.mSynchronous = false;
    req.mThread = mResourceMemoryThread;
    req.mDelegate = &mClearAllCachesFn;
    req.mName = "ClearAllCaches";
    req.mData_8 = false;
    req.mData_c = -1;
    req.mData_mStr = arena->getHeap()->getName();

    util::TaskMgrRequest task_mgr_request;
    task_mgr_request.request = &req;
    mResourceMemoryTaskMgr->submitRequest(task_mgr_request);
}

bool ResourceMgrTask::isDefragDone() const {
    return mTask2->getStatus() == util::Task::Status::PostFinishCallback;
}

f32 ResourceMgrTask::getDefragProgress() const {
    auto lock = sead::makeScopedLock(mArenasCS);
    if (_4c8 == -1)
        return 0;
    return f32(_4cc) / f32(_4c8);
}

void ResourceMgrTask::registerFactory(sead::ResourceFactory* factory,
                                      const sead::SafeString& name) {
    auto lock = sead::makeScopedLock(mFactoryCS);
    sead::ResourceMgr::instance()->registerFactory(factory, name);
}

void ResourceMgrTask::unregisterFactory(sead::ResourceFactory* factory) {
    auto lock = sead::makeScopedLock(mFactoryCS);
    sead::ResourceMgr::instance()->unregisterFactory(factory);
}

void ResourceMgrTask::cancelTasks() {
    stubbedLogFunction();
    stubbedLogFunction();
    mMovableMemoryThread->cancelTasks(u8(LaneId::_1));
    mMovableMemoryThread->cancelTasks(u8(LaneId::_0));
    mMovableMemoryThread->cancelTasks(u8(LaneId::_3));
    mMovableMemoryThread->cancelTasks(u8(LaneId::_4));

    stubbedLogFunction();
    stubbedLogFunction();
    mResourceControlThread->cancelTasks(u8(LaneId::_0));
    mResourceControlThread->cancelTasks(u8(LaneId::_1));
    mResourceControlThread->cancelTasks(u8(LaneId::_2));

    stubbedLogFunction();
    stubbedLogFunction();
    mResourceMemoryThread->cancelTasks(u8(LaneId::_1));
    mResourceMemoryThread->cancelTasks(u8(LaneId::_2));
    mResourceMemoryThread->cancelTasks(u8(LaneId::_3));
    mResourceMemoryThread->cancelTasks(u8(LaneId::_4));
    mResourceMemoryThread->cancelTasks(u8(LaneId::_5));
    mResourceMemoryThread->cancelTasks(u8(LaneId::_6));

    stubbedLogFunction();
    stubbedLogFunction();
    mResourceLoadingThread->cancelTasks(u8(LaneId::_0));
    mResourceLoadingThread->cancelTasks(u8(LaneId::_1));
    mResourceLoadingThread->cancelTasks(u8(LaneId::_2));

    stubbedLogFunction();
    stubbedLogFunction();
}

void ResourceMgrTask::waitForTaskQueuesToEmpty() {
    if (mResourceControlThread->isPaused() || mResourceMemoryThread->isPaused() ||
        mResourceLoadingThread->isPaused()) {
        return;
    }

    mResourceControlThread->waitForQueueToEmpty();
    mResourceMemoryThread->waitForQueueToEmpty();
    mResourceLoadingThread->waitForQueueToEmpty();
}

s32 ResourceMgrTask::getNumActiveTasksOnResLoadingThread() const {
    return mResourceLoadingThread->getNumActiveTasks();
}

OffsetReadFileDevice* ResourceMgrTask::getOffsetReadFileDevice() const {
    return mOffsetReadFileDevice;
}

sead::ArchiveFileDevice* ResourceMgrTask::getArchiveFileDev1() {
    return &mArchiveFileDev1;
}

void ResourceMgrTask::controlField9c0d88(bool off) {
    if (off) {
        _9c0d88.decrement();
        static_cast<void>(_9c0d88.load());
    } else {
        _9c0d88.increment();
    }

    mFlags.change(Flag::_400, _9c0d88 <= 0);
}

void ResourceMgrTask::setFlag2000Or5000(s32 type) {
    if (type != 0) {
        mFlags.set(Flag::_1000);
        mFlags.set(Flag::_4000);
        stubbedLogFunction();
    } else {
        mFlags.set(Flag::_2000);
    }
}

void ResourceMgrTask::resetFlag20000() {
    mFlags.reset(Flag::_20000);
}

bool ResourceMgrTask::isFlag4Set() const {
    return mFlags.isOn(Flag::_4);
}

void ResourceMgrTask::registerUnit(ResourceUnit* unit) {
    auto lock = sead::makeScopedLock(mUnitsCS);
    mUnits.pushBack(unit);
    if (res::returnFalse())
        stubbedLogFunction();
}

void ResourceMgrTask::deregisterUnit(ResourceUnit* unit) {
    auto lock = sead::makeScopedLock(mUnitsCS);
    if (unit->isLinkedToResourceMgr()) {
        mUnits.erase(unit);
        if (res::returnFalse())
            stubbedLogFunction();
    }
}

void ResourceMgrTask::requestClearCache(ResourceUnit** p_unit, util::Task* task) {
    if (!p_unit || !*p_unit || !(*p_unit)->isStatusFlag8000Set()) {
        stubbedLogFunction();
        return;
    }

    if ((*p_unit)->getRefCount() > 0 || (*p_unit)->isStatus1() || (*p_unit)->isStatusFlag10000Set())
        return;

    (*p_unit)->setStatusFlag10000();

    {
        ControlTaskRequest req;
        req.mHasHandle = true;
        req.mSynchronous = false;
        req.mLaneId = u8(LaneId::_4);
        req.mThread = mResourceControlThread;
        req.mDelegate = &mUnitClearCacheForSyncFn.fn;
        req.mPostRunCallback = &mUnitClearCacheForSyncFn.cb;
        req.mUserData = *p_unit;
        req.mName = "ClearCache";
        if (task)
            task->submitRequest(req);
        else
            (*p_unit)->mTask3.submitRequest(req);
    }
    *p_unit = nullptr;
}

void ResourceMgrTask::requestClearCacheForSync(ResourceUnit** p_unit, bool clear_immediately,
                                               bool delete_immediately) {
    if (!p_unit || !*p_unit || !(*p_unit)->isStatusFlag8000Set()) {
        goto fail;
    }

    if ((*p_unit)->isStatus1() || !(*p_unit)->mCache || (*p_unit)->getRefCount() > 0)
        return;

    if ((*p_unit)->isStatusFlag10000Set()) {
    fail:
        stubbedLogFunction();
        return;
    }

    (*p_unit)->setStatusFlag10000();

    if (clear_immediately) {
        (*p_unit)->removeTask3FromQueue();
        (*p_unit)->clearCacheForSync(true);
        (*p_unit)->clearCache(nullptr);
        static_cast<void>((*p_unit)->getStatus());
        ResourceUnit* ptr = *p_unit;
        if (delete_immediately)
            deleteUnit(ptr, true);
        else
            requestDeleteUnit(&ptr);

    } else {
        ControlTaskRequest req;
        req.mHasHandle = true;
        req.mSynchronous = true;
        req.mLaneId = u8(LaneId::_4);
        req.mThread = mResourceControlThread;
        req.mDelegate = &mUnitClearCacheForSyncFn.fn;
        req.mUserData = *p_unit;
        req.mName = "ClearCache(ForSync)";
        (*p_unit)->mTask3.submitRequest(req);
    }
    *p_unit = nullptr;
}

#ifdef MATCHING_HACK_NX_CLANG
[[gnu::noinline]]
#endif
void ResourceMgrTask::deleteUnit(ResourceUnit*& unit, bool sync) {
    if (!unit)
        return;

    const bool immediate = mResourceControlThread->isPaused() || sync;
    if (!immediate && res::returnFalse())
        stubbedLogFunction();

    unit->unloadArchiveRes();

    if (immediate) {
        mUnitPool.freeForSync(unit);
        unit = nullptr;
    } else {
        mUnitPool.free(unit);
        unit = nullptr;
        if (res::returnFalse())
            stubbedLogFunction();
    }
}

void ResourceMgrTask::requestDeleteUnit(ResourceUnit** p_unit) {
    if (!p_unit || !*p_unit) {
        stubbedLogFunction();
        return;
    }

    if (mResourceControlThread->isPaused()) {
        if (res::returnFalse())
            stubbedLogFunction();

        deleteUnit(*p_unit, false);

    } else {
        ControlTaskRequest req;
        req.mHasHandle = false;
        req.mSynchronous = false;
        req.mLaneId = u8(LaneId::_5);
        req.mThread = mResourceControlThread;
        req.mDelegate = &mUnitDeleteFn;
        req.mName = "DeleteUnit";
        req.mUserData = *p_unit;
        (*p_unit)->mTask1.submitRequest(req);
    }

    *p_unit = nullptr;
}

bool ResourceMgrTask::canUseSdCard() const {
    return false;
}

bool ResourceMgrTask::isHostPath(const sead::SafeString&) const {
    return false;
}

bool ResourceMgrTask::calc_(void*) {
    if (mCacheControlFlags.testAndClear(CacheControlFlag::ClearAllCachesRequested)) {
        MemoryTaskRequest req;
        req.mLaneId = u8(LaneId::_9);
        req.mHasHandle = true;
        req.mSynchronous = true;
        req.mThread = mResourceMemoryThread;
        req.mDelegate = &mClearAllCachesFn;
        req.mName = "ClearAllCaches";
        req.mData_8 = false;
        req.mData_c = -1;

        util::TaskMgrRequest request;
        request.request = &req;
        mResourceMemoryTaskMgr->submitRequest(request);
        mTexHandleMgr->clearAllCache();
        stubbedLogFunction();
    }
    clearUnits_();
    return true;
}

bool ResourceMgrTask::callSystemCalc_(void* userdata) {
    systemCalc_();
    return true;
}

bool ResourceMgrTask::dropSFromExtensionIfNeeded(const sead::SafeString& path,
                                                 sead::BufferedSafeString& new_path, s32 dot_idx,
                                                 const sead::SafeString& extension) const {
    if (extension == "sbfevfl" || extension == "sbcamanim" || extension == "sbarslist") {
        new_path.copyAtWithTerminate(0, path, dot_idx);
        new_path.appendWithFormat(".%s", &extension.at(1));
        return true;
    }
    return mExtensions2.binarySearch(&extension) != -1;
}

void ResourceMgrTask::unloadSeadResource(sead::Resource* resource) {
    if (res::returnFalse())
        stubbedLogFunction();

    sead::ResourceMgr::instance()->unload(resource);
    stubbedLogFunction();

    if (res::returnFalse())
        stubbedLogFunction();
}

u32 ResourceMgrTask::getResourceSize(const sead::SafeString& name, void* userdata) const {
    if (!userdata)
        return mResourceInfoContainer.getResourceSize(name);

    mFileDevicePrefixesLock.readLock();

    for (const auto& entry : mFileDevicePrefixes) {
        if (entry.getUserData() == userdata) {
            const u32 size = mResourceInfoContainer.getResourceSize(entry.getPrefix(), name);
            if (size == 0 && entry.getField28())
                break;

            mFileDevicePrefixesLock.readUnlock();
            return size;
        }
    }

    mFileDevicePrefixesLock.readUnlock();
    return mResourceInfoContainer.getResourceSize(name);
}

void ResourceMgrTask::registerFileDevicePrefix(FileDevicePrefix& prefix) {
    mFileDevicePrefixesLock.writeLock();
    mFileDevicePrefixes.pushBack(&prefix);
    mFileDevicePrefixesLock.writeUnlock();
}

void ResourceMgrTask::deregisterFileDevicePrefix(FileDevicePrefix& prefix) {
    mFileDevicePrefixesLock.writeLock();
    mFileDevicePrefixes.erase(&prefix);
    mFileDevicePrefixesLock.writeUnlock();
}

void ResourceMgrTask::callStubbedFunctionOnArenas() {
    auto lock = sead::makeScopedLock(mArenasCS);
    for (OverlayArena& arena : mArenas) {
        if (arena.isFlag8Set())
            arena.stubbed();
    }
}

void ResourceMgrTask::updateResourceArenasFlag8() {
    mArenaForResourceS.updateFlag8(false);
    mArenaForResourceL.updateFlag8(false);
}

// NON_MATCHING: branching
sead::Heap* ResourceMgrTask::makeHeapForUnit(const MakeHeapArg& arg) {
    const auto heap_size = arg.heap_size;
    const auto path = arg.path;

    OverlayArena* arena = arg.arena;
    if (!arena) {
        if (heap_size > 0x80000)
            arena = &mArenaForResourceL;
        else
            arena = &mArenaForResourceS;
    }

    sead::Heap* const heap =
        arena->makeDualHeap(heap_size, path, sead::Heap::cHeapDirection_Forward, arg.unit, false);

    if (!heap) {
        static_cast<void>(arena->isFlag10Set());
        *arg.out_arena1 = arena;
        return nullptr;
    }

    if (arg.out_arena2 == nullptr)
        return heap;

    *arg.out_arena1 = arena;
    *arg.out_arena2 = arena;
    return heap;
}

ResourceUnit* ResourceMgrTask::clearCachesAndGetUnit(const GetUnitArg& arg) {
    auto* unit = mUnitPool.tryAlloc();

    if (!unit) {
        util::TaskQueueLock lock;

        auto* queue = mResourceControlThread->getTaskQueue();
        auto it = queue->activeTasksRobustBegin(&lock);
        const auto end = queue->activeTasksRobustEnd();

        while (it != end && it->getLaneId() >= u8(ResourceMgrTask::LaneId::_5)) {
            it->removeFromQueue2();
            it->processOnCurrentThreadDirectly(mResourceControlThread);
            ++it;
        }

        unit = mUnitPool.tryAlloc();
    }

    if (!unit) {
        ClearCachesTaskData data;
        data._8 = true;
        data._c = 100;
        util::TaskRequest req;
        req.mLaneId = u8(LaneId::_8);
        req.mHasHandle = true;
        req.mSynchronous = true;
        req.mThread = mResourceMemoryThread;
        req.mDelegate = &mClearCachesFn;
        req.mUserData = &data;
        req.mName = "ClearCaches";
        mTask3->submitRequest(req);

        unit = mUnitPool.tryAlloc();
    }

    if (!unit) {
        util::TaskQueueLock lock;

        auto* queue = mResourceControlThread->getTaskQueue();
        auto it = queue->activeTasksRobustBegin(&lock);
        const auto end = queue->activeTasksRobustEnd();

        while (it != end && it->getLaneId() >= u8(ResourceMgrTask::LaneId::_5)) {
            it->removeFromQueue2();
            it->processOnCurrentThreadDirectly(mResourceControlThread);
            ++it;
        }

        unit = mUnitPool.alloc();
    }

    if (!unit->init(*arg.unit_init_arg))
        return nullptr;

    return unit;
}

void ResourceMgrTask::setActorCreateInitializerThreads(
    const SetActorCreateInitializerThreadsArg& arg) {
    mFlags.set(Flag::_8);
    mActorCreateInitializerThreads = arg.threads;
    stubbedLogFunction();
}

void ResourceMgrTask::clearActorCreateInitializerThreads() {
    mFlags.reset(Flag::_8);
    mActorCreateInitializerThreads = nullptr;
    stubbedLogFunction();
}

void ResourceMgrTask::pauseThreads() {
    stubbedLogFunction();
    mMovableMemoryThread->pauseAndWaitForAck();
    mMovableMemoryThread->cancelTasks(3);
    mResourceControlThread->pauseAndWaitForAck();
    mResourceMemoryThread->pauseAndWaitForAck();
    mResourceLoadingThread->pauseAndWaitForAck();
    stubbedLogFunction();
}

void ResourceMgrTask::resumeThreads() {
    stubbedLogFunction();
    mResourceLoadingThread->resume();
    mResourceMemoryThread->resume();
    mResourceControlThread->resume();
    mMovableMemoryThread->resume();
}

sead::SZSDecompressor* ResourceMgrTask::getSzsDecompressor() {
    mSzsDecompressorCS.lock();
    sead::SZSDecompressor* ptr = nullptr;
    OverlayArenaSystem::instance()->getSzsDecompressor(&ptr);
    return ptr;
}

void ResourceMgrTask::unlockSzsDecompressorCS() {
    mSzsDecompressorCS.unlock();
}

bool ResourceMgrTask::getUncompressedSize(u32* size, const sead::SafeString& path,
                                          sead::FileDevice* device) const {
    auto lock = sead::makeScopedLock(mSzsDecompressorCS);

    if (!device)
        device = mSeadMainFileDevice;

    sead::FileHandle handle;
    if (!device->tryOpen(&handle, path, sead::FileDevice::cFileOpenFlag_ReadOnly)) {
        stubbedLogFunction();
        return false;
    }

    u32 read_size = 0;
    handle.tryRead(&read_size, mOffsetReadBuf, 0x10);
    *size = sead::Mathu::roundUpPow2(sead::SZSDecompressor::getDecompSize(mOffsetReadBuf), 32);
    return true;
}

void ResourceMgrTask::clearUnits_() {
    const auto lock = sead::makeScopedLock(mUnitsCS);

    const int num_units = mUnits.size();
    if (num_units != 0 && returnFalse())
        stubbedLogFunction();

    for (auto& unit_ref : mUnits.robustRange()) {
        ResourceUnit* unit = &unit_ref;
        if (unit->mTask3.canSubmitRequest()) {
            mUnits.erase(unit);
            requestClearCache(&unit);
        }
    }

    if (mUnits.size() != 0 || (num_units != 0 && returnFalse()))
        stubbedLogFunction();
}

void ResourceMgrTask::systemCalc_() {
    if (mTask1->canSubmitRequest()) {
        util::TaskRequest request;
        request.mSynchronous = false;
        request.mHasHandle = false;
        request.mLaneId = u8(LaneId::_6);
        request.mThread = mResourceControlThread;
        request.mName = "res::System::calc";
        mTask1->submitRequest(request);
    }

    mTexHandleMgr->preCalc();
    updateCompaction();
    mTexHandleMgr->calc();
}

// NON_MATCHING: reordering
void ResourceMgrTask::setCompactionStopped(bool stopped) {
    u32 old_counter;
    if (stopped)
        old_counter = mCompactionCounter.decrement();
    else
        old_counter = mCompactionCounter.increment();

    stubbedLogFunction();
    if (mCompactionCounter == 0 || old_counter == 0)
        stubbedLogFunction();
}

bool ResourceMgrTask::isCompactionStopped() const {
    return mCompactionCounter == 0;
}

void ResourceMgrTask::requestCalc() {
    frm::WorkerSupportThreadMgr::instance()->submitRequest(2, &mSystemCalcFn);
}

void ResourceMgrTask::waitForCalc() {
    frm::WorkerSupportThreadMgr::instance()->waitForTask(2);
}

bool ResourceMgrTask::initTempResourceLoader(TempResourceLoader* loader,
                                             TempResourceLoader::InitArg& arg) {
    arg.work = mTexHandleMgr->getArchiveWork();
    return loader->init(arg);
}

bool ResourceMgrTask::returnTrue1() {
    return true;
}

void ResourceMgrTask::clearCacheWithFileExtension(const sead::SafeString& extension) {
    stubbedLogFunction();
    const s32 idx = getCacheIdx(extension);
    stubbedLogFunction();
    mCaches[idx]->eraseUnits();
}

void ResourceMgrTask::clearAllCachesSynchronously(OverlayArena* arena) {
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_5));
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_4));
    mResourceControlThread->getTaskQueue()->waitForLaneToEmpty(u8(LaneId::_3));

    MemoryTaskRequest req;
    req.mLaneId = u8(LaneId::_9);
    req.mHasHandle = true;
    req.mSynchronous = true;
    req.mThread = mResourceMemoryThread;
    req.mDelegate = &mClearAllCachesFn;
    req.mName = "ClearAllCaches";
    req.mData_8 = false;
    req.mData_c = -1;
    req.mData_mStr = arena->getHeap()->getName();

    util::TaskMgrRequest task_mgr_request;
    task_mgr_request.request = &req;
    mResourceMemoryTaskMgr->submitRequest(task_mgr_request);
}

bool ResourceMgrTask::returnTrue() {
    return true;
}

void ResourceMgrTask::removeOverlayArena(OverlayArena* arena) {
    mTask.removeFromQueue();

    auto lock = sead::makeScopedLock(mArenasCS);
    mArenaIdx = 0;
    if (mArenas.isNodeLinked(arena)) {
        mArenas.erase(arena);
        stubbedLogFunction();
    }
}

void ResourceMgrTask::setPack(Handle* pack) {
    mPackHandle = pack;
    res::stubbedLogFunction();
}

}  // namespace ksys::res
