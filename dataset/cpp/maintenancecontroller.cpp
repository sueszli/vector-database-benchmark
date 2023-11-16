// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "maintenancecontroller.h"
#include "maintenancejobrunner.h"
#include "i_blockable_maintenance_job.h"
#include <vespa/searchcorespi/index/i_thread_service.h>
#include <vespa/searchcore/proton/common/scheduledexecutor.h>
#include <vespa/vespalib/util/lambdatask.h>
#include <vespa/vespalib/util/thread.h>

#include <vespa/log/log.h>
LOG_SETUP(".proton.server.maintenancecontroller");

using document::BucketId;
using vespalib::Executor;
using vespalib::MonitoredRefCount;
using vespalib::makeLambdaTask;
using vespalib::thread::as_zu;

namespace proton {

namespace {

class JobWrapperTask : public Executor::Task
{
private:
    MaintenanceJobRunner *_job;
public:
    JobWrapperTask(MaintenanceJobRunner *job) : _job(job) {}
    void run() override { _job->run(); }
};

}

MaintenanceController::MaintenanceController(FNET_Transport & transport,
                                             ISyncableThreadService& masterThread,
                                             MonitoredRefCount& refCount,
                                             const DocTypeName& docTypeName)
    : _masterThread(masterThread),
      _refCount(refCount),
      _readySubDB(),
      _remSubDB(),
      _notReadySubDB(),
      _periodicTimer(std::make_unique<ScheduledExecutor>(transport)),
      _state(State::INITIALIZING),
      _docTypeName(docTypeName),
      _jobs(),
      _jobsLock()
{ }

MaintenanceController::~MaintenanceController()
{
    kill();
}

void
MaintenanceController::registerJob(IMaintenanceJob::UP job)
{
    // Called by master write thread
    Guard guard(_jobsLock);
    _jobs.push_back(std::make_shared<MaintenanceJobRunner>(_masterThread, std::move(job)));
}

void
MaintenanceController::killJobs()
{
    if (_state == State::STARTED) {
        _state = State::PAUSED;
    }
    // Called by master write thread
    assert(_masterThread.isCurrentThread());
    LOG(debug, "killJobs(): threadId=%zu", as_zu(std::this_thread::get_id()));
    _periodicTaskHandles.clear();
    // No need to take _jobsLock as modification of _jobs also happens in master write thread.
    for (auto &job : _jobs) {
        job->stop(); // Make sure no more tasks are added to the executor
    }
    JobList tmpJobs;
    {
        Guard guard(_jobsLock);
        tmpJobs.swap(_jobs);
    }
    // Hold jobs until existing tasks have been drained
    _masterThread.execute(makeLambdaTask([this, jobs=std::move(tmpJobs)]() {
        performHoldJobs(std::move(jobs));
    }));
}

void
MaintenanceController::updateMetrics(DocumentDBTaggedMetrics & metrics)
{
    Guard guard(_jobsLock);
    for (auto &job : _jobs) {
        job->getJob().updateMetrics(metrics); // Make sure no more tasks are added to the executor
    }
}

void
MaintenanceController::performHoldJobs(JobList jobs)
{
    // Called by master write thread
    LOG(debug, "performHoldJobs(): threadId=%zu", as_zu(std::this_thread::get_id()));
    (void) jobs;
}

void
MaintenanceController::stop()
{
    assert(!_masterThread.isCurrentThread());
    _masterThread.execute(makeLambdaTask([this]() { _state = State::STOPPING; killJobs(); }));
    _masterThread.sync();  // Wait for killJobs()
    _masterThread.sync();  // Wait for already scheduled maintenance jobs and performHoldJobs
}

searchcorespi::index::IThreadService &
MaintenanceController::masterThread() {
    return _masterThread;
}

void
MaintenanceController::kill()
{
    stop();
    _readySubDB.clear();
    _remSubDB.clear();
    _notReadySubDB.clear();
}

void
MaintenanceController::start()
{
    // Called by master write thread
    assert(_state == State::INITIALIZING);
    _state = State::STARTED;
    restart();
}


void
MaintenanceController::restart()
{
    // Called by master write thread
    if (!getStarted() || getStopping() || !_readySubDB.valid()) {
        return;
    }
    addJobsToPeriodicTimer();
}

void
MaintenanceController::addJobsToPeriodicTimer()
{
    _periodicTaskHandles.clear();
    // No need to take _jobsLock as modification of _jobs also happens in master write thread.
    for (const auto &jw : _jobs) {
        const IMaintenanceJob &job = jw->getJob();
        LOG(debug, "addJobsToPeriodicTimer(): docType='%s', job.name='%s', job.delay=%f, job.interval=%f",
            _docTypeName.getName().c_str(), job.getName().c_str(), vespalib::to_s(job.getDelay()),
            vespalib::to_s(job.getInterval()));
        if (job.getInterval() == vespalib::duration::zero()) {
            jw->run();
            continue;
        }
        _periodicTaskHandles.push_back(_periodicTimer->scheduleAtFixedRate(std::make_unique<JobWrapperTask>(jw.get()),
                                                                           job.getDelay(), job.getInterval()));
    }

}

void
MaintenanceController::newConfig()
{
    // Called by master write thread
    restart();
}

namespace {

void
assert_equal_meta_store_instances(const MaintenanceDocumentSubDB& old_db,
                                  const MaintenanceDocumentSubDB& new_db)
{
    if (old_db.valid() && new_db.valid()) {
        assert(old_db.meta_store().get() == new_db.meta_store().get());
    }
}

}

void
MaintenanceController::syncSubDBs(const MaintenanceDocumentSubDB &readySubDB,
                                  const MaintenanceDocumentSubDB &remSubDB,
                                  const MaintenanceDocumentSubDB &notReadySubDB)
{
    // Called by master write thread
    bool oldValid = _readySubDB.valid();
    assert(readySubDB.valid());
    assert(remSubDB.valid());
    // Document meta store instances should not change. Maintenance jobs depend on this fact.
    assert_equal_meta_store_instances(_readySubDB, readySubDB);
    assert_equal_meta_store_instances(_remSubDB, remSubDB);
    assert_equal_meta_store_instances(_notReadySubDB, notReadySubDB);
    _readySubDB = readySubDB;
    _remSubDB = remSubDB;
    _notReadySubDB = notReadySubDB;
    if (!oldValid && getStarted()) {
        restart();
    }
}

} // namespace proton
