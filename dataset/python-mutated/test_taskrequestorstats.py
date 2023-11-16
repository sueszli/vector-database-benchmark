from unittest import TestCase
from unittest.mock import Mock, patch
from pydispatch import dispatcher
from golem import testutils
from golem.task.taskrequestorstats import TaskInfo, TaskMsg, RequestorTaskStats, logger, CurrentStats, TaskStats, EMPTY_TASK_STATS, FinishedTasksStats, FinishedTasksSummary, RequestorTaskStatsManager, EMPTY_CURRENT_STATS, EMPTY_FINISHED_STATS, AggregateTaskStats, RequestorAggregateStatsManager
from golem.task.taskstate import TaskStatus, Operation, TaskOp, SubtaskOp, OtherOp, SubtaskStatus, TaskState
from golem.testutils import DatabaseFixture
from golem.tools.assertlogs import LogTestCase
from tests.factories.task import taskstate as taskstate_factory

class TestTaskInfo(TestCase, testutils.PEP8MixIn):
    PEP8_FILES = ['golem/task/taskrequestorstats.py', 'tests/golem/task/test_taskrequestorstats.py']

    def test_taskinfo_creation(self):
        if False:
            i = 10
            return i + 15
        ti = TaskInfo()
        self.assertIsNotNone(ti, 'TaskInfo() returned None')
        self.assertEqual(ti.latest_status, TaskStatus.notStarted, 'Newly created TaskInfo should haveTaskStatus.notStarted status')
        self.assertEqual(ti.subtask_count(), 0, 'Newly created TaskInfo should have no subtasks')
        self.assertEqual(ti.want_to_compute_count(), 0, 'Newly created TaskInfo should have not received any want to compute offers yet')

    def test_task_with_one_subtask(self):
        if False:
            return 10
        ti = TaskInfo()
        tm = TaskMsg(ts=1.0, op=TaskOp.CREATED)
        ti.got_task_message(tm, TaskStatus.waiting)
        self.assertEqual(ti.latest_status, TaskStatus.waiting, 'TaskInfo should store the latest status supplied')
        self.assertEqual(ti.subtask_count(), 0, 'TaskInfo should have no subtasks at this point')
        self.assertEqual(ti.want_to_compute_count(), 0, 'TaskInfo should have not received any want to compute offers yet')
        tm = TaskMsg(ts=1.5, op=TaskOp.STARTED)
        ti.got_task_message(tm, TaskStatus.starting)
        ti.got_want_to_compute()
        self.assertEqual(ti.want_to_compute_count(), 1, 'TaskInfo should have received one want to compute offer already')
        tm = TaskMsg(ts=2.0, op=SubtaskOp.ASSIGNED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.starting)
        self.assertEqual(ti.subtask_count(), 1, 'TaskInfo should have one subtask at this point')
        self.assertEqual(ti.collected_results_count(), 0, 'No results should have been collected yet')
        self.assertEqual(ti.verified_results_count(), 0, 'No results should have been verified yet')
        self.assertEqual(ti.not_accepted_results_count(), 0, 'No results should have been not accepted yet')
        self.assertEqual(ti.timeout_count(), 0, 'No results should have timed out yet')
        self.assertEqual(ti.not_downloaded_count(), 0, 'No results should have had problems w/download yet')
        self.assertGreaterEqual(ti.total_time(), 1.0, 'Total time should be larger than 1.0 at this point since the task is not finished yet')
        self.assertFalse(ti.had_failures_or_timeouts(), 'No timeouts nor failures expected so far')
        self.assertFalse(ti.is_completed(), 'Task should not be considered done')
        self.assertEqual(ti.in_progress_subtasks_count(), 1, 'One subtask should be in progress')
        tm = TaskMsg(ts=3.0, op=SubtaskOp.RESULT_DOWNLOADING)
        ti.got_subtask_message('st1', tm, SubtaskStatus.downloading)
        self.assertEqual(ti.in_progress_subtasks_count(), 1, 'One subtask should still be in progress')
        self.assertEqual(ti.not_downloaded_count(), 1, 'Results of one subtask are being downloaded now')
        tm = TaskMsg(ts=4.0, op=SubtaskOp.FINISHED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.finished)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertEqual(ti.subtask_count(), 1, 'TaskInfo should have one subtask at this point')
        self.assertFalse(ti.is_completed(), 'Task should not be considered done')
        self.assertEqual(ti.collected_results_count(), 1, 'One result should have been collected already')
        self.assertEqual(ti.verified_results_count(), 1, 'One result should have been verified already')
        self.assertEqual(ti.not_accepted_results_count(), 0, 'No results should have been not accepted yet')
        self.assertEqual(ti.timeout_count(), 0, 'No results should have timed out yet')
        self.assertEqual(ti.not_downloaded_count(), 0, 'No results should have had problems w/download yet')
        tm = TaskMsg(ts=5.0, op=TaskOp.FINISHED)
        ti.got_task_message(tm, TaskStatus.finished)
        self.assertTrue(ti.is_completed(), 'Task should be considered done now')
        self.assertEqual(ti.total_time(), 4.0, 'Total time should equal 4.0 at this point')

    @staticmethod
    def _create_task_with_single_subtask(subtask_name='st1'):
        if False:
            for i in range(10):
                print('nop')
        ti = TaskInfo()
        tm = TaskMsg(ts=1.0, op=TaskOp.CREATED)
        ti.got_task_message(tm, TaskStatus.waiting)
        tm = TaskMsg(ts=2.0, op=SubtaskOp.ASSIGNED)
        ti.got_subtask_message(subtask_name, tm, SubtaskStatus.starting)
        return ti

    def test_task_with_two_subtasks(self):
        if False:
            i = 10
            return i + 15
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=SubtaskOp.ASSIGNED)
        ti.got_subtask_message('st2', tm, SubtaskStatus.starting)
        self.assertEqual(ti.subtask_count(), 2, 'TaskInfo should have two subtasks at this point')
        self.assertEqual(ti.in_progress_subtasks_count(), 2, 'Both subtasks should be in progress')
        tm = TaskMsg(ts=4.0, op=SubtaskOp.RESULT_DOWNLOADING)
        ti.got_subtask_message('st1', tm, SubtaskStatus.downloading)
        tm = TaskMsg(ts=5.0, op=SubtaskOp.FINISHED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.finished)
        self.assertEqual(ti.in_progress_subtasks_count(), 1, 'One subtask should still be in progress')
        self.assertEqual(ti.not_downloaded_count(), 0, 'No downloads should be in progress')
        self.assertEqual(ti.subtask_count(), 2, 'TaskInfo should still have two subtasks at this point')
        tm = TaskMsg(ts=6.0, op=SubtaskOp.RESULT_DOWNLOADING)
        ti.got_subtask_message('st2', tm, SubtaskStatus.downloading)
        tm = TaskMsg(ts=7.0, op=SubtaskOp.FINISHED)
        ti.got_subtask_message('st2', tm, SubtaskStatus.finished)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'One subtask should still be in progress')
        self.assertEqual(ti.not_downloaded_count(), 0, 'No downloads should be in progress')
        self.assertEqual(ti.subtask_count(), 2, 'TaskInfo should still have two subtasks at this point')
        self.assertFalse(ti.had_failures_or_timeouts(), 'Everything wenth smoothly so no failures were expected')
        self.assertEqual(ti.verified_results_count(), 2, 'Both result should have been verified already')

    def test_task_with_various_problems(self):
        if False:
            for i in range(10):
                print('nop')
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=SubtaskOp.TIMEOUT)
        ti.got_subtask_message('st1', tm, SubtaskStatus.failure)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertEqual(ti.timeout_count(), 1, 'One subtask should have timed out')
        self.assertTrue(ti.had_failures_or_timeouts(), 'One subtask should have timed out')
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=SubtaskOp.NOT_ACCEPTED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.failure)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertEqual(ti.not_accepted_results_count(), 1, 'One subtask should have not been accepted')
        self.assertTrue(ti.had_failures_or_timeouts(), 'One subtask should have not been accepted')
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=SubtaskOp.FAILED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.failure)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertTrue(ti.had_failures_or_timeouts(), 'One subtask should have failed')
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=TaskOp.TIMEOUT)
        ti.got_task_message(tm, TaskStatus.timeout)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertEqual(ti.timeout_count(), 0, 'No subtask should have timed out')
        self.assertTrue(ti.had_failures_or_timeouts(), 'Whole task should have failed')

    def test_strange_case(self):
        if False:
            while True:
                i = 10
        "An unlikely scenario, but technically not impossible.\n\n        We create a task with a subtask, then we fail the subtask and restart\n        it later on. Then we check if it is considered in progress. To be\n        honest it's just for coverage.\n        "
        ti = self._create_task_with_single_subtask()
        tm = TaskMsg(ts=3.0, op=SubtaskOp.TIMEOUT)
        ti.got_subtask_message('st1', tm, SubtaskStatus.failure)
        tm = TaskMsg(ts=4.0, op=SubtaskOp.RESTARTED)
        ti.got_subtask_message('st1', tm, SubtaskStatus.restarted)
        self.assertEqual(ti.in_progress_subtasks_count(), 0, 'No subtasks should be in progress')
        self.assertTrue(ti.had_failures_or_timeouts(), 'One subtask should have failed')

class TestRequestorTaskStats(LogTestCase):

    def compare_task_stats(self, ts1, ts2):
        if False:
            return 10
        self.assertGreaterEqual(ts1.total_time, ts2.total_time)
        self.assertEqual(ts1.finished, ts2.finished)
        self.assertEqual(ts1.task_failed, ts2.task_failed)
        self.assertEqual(ts1.had_failures, ts2.had_failures)
        self.assertEqual(ts1.work_offers_cnt, ts2.work_offers_cnt)
        self.assertEqual(ts1.requested_subtasks_cnt, ts2.requested_subtasks_cnt)
        self.assertEqual(ts1.collected_results_cnt, ts2.collected_results_cnt)
        self.assertEqual(ts1.verified_results_cnt, ts2.verified_results_cnt)
        self.assertEqual(ts1.timed_out_subtasks_cnt, ts2.timed_out_subtasks_cnt)
        self.assertEqual(ts1.not_downloaded_subtasks_cnt, ts2.not_downloaded_subtasks_cnt)
        self.assertEqual(ts1.failed_subtasks_cnt, ts2.failed_subtasks_cnt)

    def test_stats_collection(self):
        if False:
            i = 10
            return i + 15
        rs = RequestorTaskStats()
        tstate = TaskState()
        tstate.status = TaskStatus.notStarted
        tstate.time_started = 0.0
        rs.on_message('task1', tstate, None, TaskOp.CREATED)
        self.assertFalse(rs.is_task_finished('task1'), 'task1 should be in progress')
        task1_ts = rs.get_task_stats('task1')
        self.compare_task_stats(task1_ts, EMPTY_TASK_STATS)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 0, 0, 0, 0, 0, 0, 0), 'There should be one task only with no information about any subtasks')
        tstate.status = TaskStatus.waiting
        rs.on_message('task1', tstate, op=TaskOp.STARTED)
        self.assertEqual(cs, CurrentStats(1, 0, 0, 0, 0, 0, 0, 0, 0), 'There should be one task only with no information about any subtasks')
        with self.assertLogs(logger, level='INFO') as log:
            rs.on_message('task1', tstate, op=TaskOp.WORK_OFFER_RECEIVED)
            self.assertTrue(any(('Received work offers' in line for line in log.output)))
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 0, 0, 0, 0, 0, 0, 1), 'Got work offer now')
        tstate.subtask_states['st1'] = taskstate_factory.SubtaskState()
        sst = tstate.subtask_states['st1']
        rs.on_message('task1', tstate, 'st1', SubtaskOp.ASSIGNED)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 1, 0, 0, 0, 0, 0, 1), 'One subtask was requested so far, otherwise there should be no changes to stats')
        sst.status = SubtaskStatus.downloading
        rs.on_message('task1', tstate, 'st1', SubtaskOp.RESULT_DOWNLOADING)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 1, 0, 0, 0, 0, 0, 1), "One subtask is still in progress, and even though its results are being downloaded it's not shown in the stats")
        sst.status = SubtaskStatus.finished
        rs.on_message('task1', tstate, 'st1', SubtaskOp.FINISHED)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 1, 1, 1, 0, 0, 0, 1), 'Sole subtask was finished which means its results were collected and verified')
        rs.on_message('task1', tstate, op=OtherOp.UNEXPECTED)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 0, 1, 1, 1, 0, 0, 0, 1), 'Unexpected subtask have no influence on stats')
        tstate.status = TaskStatus.finished
        rs.on_message('task1', tstate, op=TaskOp.FINISHED)
        cs = rs.get_current_stats()
        self.assertEqual(cs, CurrentStats(1, 1, 1, 1, 1, 0, 0, 0, 1), 'The only task is now finished')
        self.assertTrue(rs.is_task_finished('task1'), 'A task should be finished now')
        with self.assertNoLogs(logger, level='INFO'):
            rs.on_message('task1', tstate, op=TaskOp.WORK_OFFER_RECEIVED)

    @staticmethod
    def create_task_and_taskstate(rs, name):
        if False:
            for i in range(10):
                print('nop')
        tstate = TaskState()
        tstate.status = TaskStatus.notStarted
        tstate.time_started = 0.0
        rs.on_message(name, tstate, op=TaskOp.CREATED)
        tstate.status = TaskStatus.waiting
        rs.on_message(name, tstate, op=TaskOp.STARTED)
        rs.on_message(name, tstate, op=TaskOp.WORK_OFFER_RECEIVED)
        return tstate

    @staticmethod
    def add_subtask(rs, task, tstate, subtask):
        if False:
            return 10
        tstate.subtask_states[subtask] = taskstate_factory.SubtaskState()
        rs.on_message(task, tstate, subtask, SubtaskOp.ASSIGNED)

    @staticmethod
    def finish_subtask(rs, task, tstate, subtask):
        if False:
            i = 10
            return i + 15
        sst = tstate.subtask_states[subtask]
        sst.status = SubtaskStatus.downloading
        rs.on_message(task, tstate, subtask, SubtaskOp.RESULT_DOWNLOADING)
        sst.status = SubtaskStatus.finished
        rs.on_message(task, tstate, subtask, SubtaskOp.FINISHED)

    @staticmethod
    def finish_task(rs, task, tstate):
        if False:
            while True:
                i = 10
        tstate.status = TaskStatus.finished
        rs.on_message(task, tstate, op=TaskOp.FINISHED)

    def test_multiple_tasks(self):
        if False:
            while True:
                i = 10
        rs = RequestorTaskStats()
        ts1 = self.create_task_and_taskstate(rs, 'task1')
        self.add_subtask(rs, 'task1', ts1, 'st1.1')
        self.add_subtask(rs, 'task1', ts1, 'st1.2')
        ts2 = self.create_task_and_taskstate(rs, 'task2')
        self.add_subtask(rs, 'task2', ts2, 'st2.1')
        self.assertFalse(rs.is_task_finished('task1'), 'task1 is still active')
        self.assertFalse(rs.is_task_finished('task2'), 'task2 is still active')
        self.assertEqual(rs.get_current_stats(), CurrentStats(2, 0, 3, 0, 0, 0, 0, 0, 2), 'Two tasks should be in progress, with 3 subtasks requested')
        self.finish_subtask(rs, 'task1', ts1, 'st1.1')
        self.assertFalse(rs.is_task_finished('task1'), 'task1 is still active')
        self.assertFalse(rs.is_task_finished('task2'), 'task2 is still active')
        self.assertEqual(rs.get_current_stats(), CurrentStats(2, 0, 3, 1, 1, 0, 0, 0, 2), 'Two tasks should be in progress, with 3 subtasks; one subtask should be collected and verified')
        self.finish_subtask(rs, 'task2', ts2, 'st2.1')
        self.assertFalse(rs.is_task_finished('task1'), 'task1 is still active')
        self.assertFalse(rs.is_task_finished('task2'), 'task2 is still active')
        self.assertEqual(rs.get_current_stats(), CurrentStats(2, 0, 3, 2, 2, 0, 0, 0, 2), 'Two tasks should be in progress, with 3 subtasks; two of the subtasks should be collected and verified')
        self.finish_task(rs, 'task2', ts2)
        self.assertFalse(rs.is_task_finished('task1'), 'task1 is still active')
        self.assertTrue(rs.is_task_finished('task2'), 'task2 is finished')
        self.assertEqual(rs.get_current_stats(), CurrentStats(2, 1, 3, 2, 2, 0, 0, 0, 2), 'One task should be in progress, with 1 subtask running and 2 finished')
        ts3 = TaskState()
        ts3.status = TaskStatus.notStarted
        ts3.time_started = 0.0
        ts3.subtask_states['st3.1'] = taskstate_factory.SubtaskState()
        ts3.subtask_states['st3.2'] = taskstate_factory.SubtaskState()
        rs.on_message('task3', ts3, op=TaskOp.RESTORED)
        self.assertFalse(rs.is_task_finished('task1'), 'task1 is still active')
        self.assertTrue(rs.is_task_finished('task2'), 'task2 is finished')
        self.assertFalse(rs.is_task_finished('task3'), 'task3 is still active')
        self.assertEqual(rs.get_current_stats(), CurrentStats(3, 1, 5, 2, 2, 0, 0, 0, 2), '2 tasks should be in progress, with 5 subtasks (2 of them are finished)')
        self.finish_subtask(rs, 'task1', ts1, 'st1.2')
        self.finish_task(rs, 'task1', ts1)
        self.finish_subtask(rs, 'task3', ts3, 'st3.2')
        self.finish_subtask(rs, 'task3', ts3, 'st3.1')
        self.finish_task(rs, 'task3', ts3)
        self.assertEqual(rs.get_current_stats(), CurrentStats(3, 3, 5, 5, 5, 0, 0, 0, 2), 'No tasks should be in progress, with all 5 subtasks collected and verified')

    def test_tasks_with_errors(self):
        if False:
            i = 10
            return i + 15
        rs = RequestorTaskStats()
        ts1 = self.create_task_and_taskstate(rs, 'task1')
        self.add_subtask(rs, 'task1', ts1, 'st1.1')
        self.add_subtask(rs, 'task1', ts1, 'st1.2')
        self.add_subtask(rs, 'task1', ts1, 'st1.3')
        self.add_subtask(rs, 'task1', ts1, 'st1.4')
        ts1.subtask_states['st1.1'].status = SubtaskStatus.downloading
        rs.on_message('task1', ts1, 'st1.1', SubtaskOp.RESULT_DOWNLOADING)
        ts1.subtask_states['st1.1'].status = SubtaskStatus.failure
        rs.on_message('task1', ts1, 'st1.1', SubtaskOp.NOT_ACCEPTED)
        stats1 = rs.get_task_stats('task1')
        self.compare_task_stats(stats1, TaskStats(False, 0.0, False, True, 1, 4, 1, 0, 0, 0, 0))
        ts1.subtask_states['st1.2'].status = SubtaskStatus.failure
        rs.on_message('task1', ts1, 'st1.2', SubtaskOp.TIMEOUT)
        stats2 = rs.get_task_stats('task1')
        self.compare_task_stats(stats2, TaskStats(False, 0.0, False, True, 1, 4, 1, 0, 1, 0, 0))
        self.assertEqual(rs.get_current_stats(), CurrentStats(1, 0, 4, 1, 0, 1, 0, 0, 1), '1 task should be in progress with 2 subtasks, one of them with timeout')
        ts1.subtask_states['st1.3'].status = SubtaskStatus.failure
        rs.on_message('task1', ts1, 'st1.3', SubtaskOp.FAILED)
        stats3 = rs.get_task_stats('task1')
        self.compare_task_stats(stats3, TaskStats(False, 0.0, False, True, 1, 4, 1, 0, 1, 0, 1))
        self.assertEqual(rs.get_current_stats(), CurrentStats(1, 0, 4, 1, 0, 1, 0, 1, 1), '1 task should be in progress with 1 subtask still running; we have one failed subtask')
        ts1.subtask_states['st1.4'].status = SubtaskStatus.downloading
        rs.on_message('task1', ts1, 'st1.4', SubtaskOp.RESULT_DOWNLOADING)
        stats4 = rs.get_task_stats('task1')
        self.compare_task_stats(stats4, TaskStats(False, 0.0, False, True, 1, 4, 1, 0, 1, 1, 1))
        self.assertEqual(rs.get_current_stats(), CurrentStats(1, 0, 4, 1, 0, 1, 0, 1, 1), '1 task should be in progress with 1 subtask')
        ts1.status = TaskStatus.timeout
        rs.on_message('task1', ts1, op=TaskOp.TIMEOUT)
        stats5 = rs.get_task_stats('task1')
        self.compare_task_stats(stats5, TaskStats(True, 0.0, True, True, 1, 4, 1, 0, 1, 1, 1))
        self.assertEqual(rs.get_current_stats(), CurrentStats(1, 1, 4, 1, 0, 1, 1, 1, 1), '1 task should be finished')

    def test_resurrected_tasks(self):
        if False:
            while True:
                i = 10
        "This should probably not happen in practice, but let's test\n        tasks that are finished and then modified.\n        "
        rs = RequestorTaskStats()
        ts1 = self.create_task_and_taskstate(rs, 'task1')
        self.add_subtask(rs, 'task1', ts1, 'st1.1')
        self.finish_subtask(rs, 'task1', ts1, 'st1.1')
        self.finish_task(rs, 'task1', ts1)
        fstats1 = rs.get_finished_stats()
        ftime1 = fstats1.finished_ok.total_time
        self.assertEqual(fstats1, FinishedTasksStats(FinishedTasksSummary(1, ftime1), FinishedTasksSummary(0, 0.0), FinishedTasksSummary(0, 0.0)))
        ts1.status = TaskStatus.timeout
        rs.on_message('task1', ts1, op=TaskOp.TIMEOUT)
        fstats2 = rs.get_finished_stats()
        ftime2 = fstats2.failed.total_time
        self.assertEqual(fstats2, FinishedTasksStats(FinishedTasksSummary(0, 0.0), FinishedTasksSummary(0, 0.0), FinishedTasksSummary(1, ftime2)))
        self.assertGreaterEqual(ftime2, ftime1, 'Time should not go back')
        ts1.status = TaskStatus.waiting
        rs.on_message('task1', ts1, op=TaskOp.RESTARTED)
        self.add_subtask(rs, 'task1', ts1, 'st1.2')
        sst = ts1.subtask_states['st1.2']
        sst.status = SubtaskStatus.downloading
        rs.on_message('task1', ts1, 'st1.2', SubtaskOp.RESULT_DOWNLOADING)
        sst.status = SubtaskStatus.failure
        rs.on_message('task1', ts1, 'st1.2', SubtaskOp.NOT_ACCEPTED)
        self.finish_task(rs, 'task1', ts1)
        fstats3 = rs.get_finished_stats()
        ftime3 = fstats3.finished_with_failures.total_time
        self.assertEqual(fstats3, FinishedTasksStats(FinishedTasksSummary(0, 0.0), FinishedTasksSummary(1, ftime3), FinishedTasksSummary(0, 0.0)))
        self.assertGreaterEqual(ftime3, ftime2, 'Time should not go back')
        ts1.status = TaskStatus.aborted
        rs.on_message('task1', ts1, op=TaskOp.ABORTED)
        fstats4 = rs.get_finished_stats()
        ftime4 = fstats4.failed.total_time
        self.assertEqual(fstats4, FinishedTasksStats(FinishedTasksSummary(0, 0.0), FinishedTasksSummary(0, 0.0), FinishedTasksSummary(1, ftime4)))
        self.assertGreaterEqual(ftime4, ftime3, 'Time should not go back')

    def test_unknown_op(self):
        if False:
            return 10
        rs = RequestorTaskStats()
        tstate = TaskState()
        tstate.status = TaskStatus.notStarted
        tstate.time_started = 0.0

        class UnknownOp(Operation):
            UNKNOWN = object()
        with self.assertLogs(logger, level='DEBUG') as log:
            rs.on_message('task1', tstate, op=UnknownOp.UNKNOWN)
            assert any(('Unknown operation' in l for l in log.output))

    def test_restore_finished_task(self):
        if False:
            print('Hello World!')
        rs = RequestorTaskStats()
        tstate = TaskState()
        tstate.status = TaskStatus.timeout
        tstate.time_started = 0.0
        with self.assertLogs(logger, level='DEBUG') as log:
            rs.on_message('task1', tstate, op=TaskOp.RESTORED)
            assert any(('Skipping completed task' in l for l in log.output))

class TestRequestorTaskStatsManager(DatabaseFixture):

    def test_empty_stats(self):
        if False:
            for i in range(10):
                print('nop')
        rtsm = RequestorTaskStatsManager()
        self.assertEqual(rtsm.get_current_stats(), EMPTY_CURRENT_STATS)
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)

    def test_single_task(self):
        if False:
            while True:
                i = 10
        rtsm = RequestorTaskStatsManager()
        tstate = TaskState()
        tstate.status = TaskStatus.notStarted
        tstate.time_started = 0.0
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id=None, op=TaskOp.CREATED)
        self.assertEqual(rtsm.get_current_stats(), CurrentStats(1, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)
        tstate.status = TaskStatus.waiting
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id=None, op=TaskOp.STARTED)
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id=None, op=TaskOp.WORK_OFFER_RECEIVED)
        self.assertEqual(rtsm.get_current_stats(), CurrentStats(1, 0, 0, 0, 0, 0, 0, 0, 1))
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)
        tstate.subtask_states['st1.1'] = taskstate_factory.SubtaskState()
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id='st1.1', op=SubtaskOp.ASSIGNED)
        self.assertEqual(rtsm.get_current_stats(), CurrentStats(1, 0, 1, 0, 0, 0, 0, 0, 1))
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)
        tstate.subtask_states['st1.1'].status = SubtaskStatus.downloading
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id='st1.1', op=SubtaskOp.RESULT_DOWNLOADING)
        tstate.subtask_states['st1.1'].status = SubtaskStatus.finished
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id='st1.1', op=SubtaskOp.FINISHED)
        self.assertEqual(rtsm.get_current_stats(), CurrentStats(1, 0, 1, 1, 1, 0, 0, 0, 1))
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)
        tstate.status = TaskStatus.finished
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id='task1', task_state=tstate, subtask_id=None, op=TaskOp.FINISHED)
        self.assertEqual(rtsm.get_current_stats(), CurrentStats(1, 1, 1, 1, 1, 0, 0, 0, 1))
        self.assertEqual(rtsm.get_finished_stats()[0][0], 1)
        self.assertGreaterEqual(rtsm.get_finished_stats()[0][1], 0.0)
        self.assertEqual(rtsm.get_finished_stats()[1], FinishedTasksSummary(0, 0.0))
        self.assertEqual(rtsm.get_finished_stats()[2], FinishedTasksSummary(0, 0.0))

    def test_bad_message(self):
        if False:
            print('Hello World!')
        rtsm = RequestorTaskStatsManager()
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id=None, task_state=TaskState())
        self.assertEqual(rtsm.get_current_stats(), EMPTY_CURRENT_STATS)
        self.assertEqual(rtsm.get_finished_stats(), EMPTY_FINISHED_STATS)

class TestAggregateTaskStats(TestCase):

    @classmethod
    def test_init(cls):
        if False:
            i = 10
            return i + 15
        stats_dict = dict(requestor_payment_cnt=1, requestor_payment_delay_avg=2.0, requestor_payment_delay_sum=3.0, requestor_subtask_timeout_mag=4, requestor_subtask_price_mag=5, requestor_velocity_timeout=6, requestor_velocity_comp_time=7)
        aggregate_stats = AggregateTaskStats(**stats_dict)
        for (key, value) in stats_dict.items():
            stats_value = getattr(aggregate_stats, key)
            assert isinstance(stats_value, type(value))
            assert stats_value == value

class TestRequestorAggregateStatsManager(TestCase):

    class MockKeeper:

        def __init__(self, *_args, **_kwargs):
            if False:
                print('Hello World!')
            self.increased_stats = dict()
            self.retrieved_stats = set()
            self.replaced_stats = dict()
            self.increase_stat = Mock(wraps=self._increase_stat)
            self.get_stats = Mock(wraps=self._get_stats)
            self.set_stat = Mock(wraps=self._set_stat)

        def _increase_stat(self, key, value):
            if False:
                i = 10
                return i + 15
            self.increased_stats[key] = value

        def _get_stats(self, key):
            if False:
                for i in range(10):
                    print('nop')
            self.retrieved_stats.add(key)
            return (0, 0)

        def _set_stat(self, key, value):
            if False:
                while True:
                    i = 10
            self.replaced_stats[key] = value

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        with patch('golem.task.taskrequestorstats.StatsKeeper', self.MockKeeper):
            self.manager = RequestorAggregateStatsManager()

    def test_on_computed_ignored_event(self):
        if False:
            print('Hello World!')
        self.manager._on_computed(event='ignored')
        assert not self.manager.keeper.increase_stat.called

    def test_on_computed_timeout(self):
        if False:
            i = 10
            return i + 15
        event_args = dict(subtask_count=10, subtask_timeout=7, subtask_price=10 ** 18, subtask_computation_time=3600.0, timed_out=True)
        self.manager._on_computed(event='finished', **event_args)
        stats = self.manager.keeper.increased_stats
        assert stats['requestor_velocity_timeout'] == event_args['subtask_computation_time']

    def test_on_computed(self):
        if False:
            i = 10
            return i + 15
        event_args = dict(subtask_count=10, subtask_timeout=7, subtask_price=10 ** 18, subtask_computation_time=3600.0)
        self.manager._on_computed(event='finished', **event_args)
        stats = self.manager.keeper.increased_stats
        assert 'requestor_velocity_timeout' not in stats
        assert stats['requestor_subtask_timeout_mag'] != 0
        assert stats['requestor_subtask_price_mag'] != 0
        assert stats['requestor_velocity_comp_time'] != 0

    def test_on_payment_ignored_event(self):
        if False:
            for i in range(10):
                print('nop')
        self.manager._on_payment(event='ignored')
        assert not self.manager.keeper.get_stats.called
        assert not self.manager.keeper.set_stat.called

    def test_on_payment(self):
        if False:
            print('Hello World!')
        kwargs = dict(delay=10, requestor_payment_cnt=13, requestor_payment_delay_sum=10 ** 3)
        self.manager._on_payment(event='confirmed', **kwargs)
        retrieved = self.manager.keeper.retrieved_stats
        replaced = self.manager.keeper.replaced_stats
        assert 'requestor_payment_cnt' in retrieved
        assert 'requestor_payment_delay_sum' in retrieved
        assert replaced['requestor_payment_cnt'] != 0
        assert replaced['requestor_payment_delay_sum'] != 0
        assert replaced['requestor_payment_delay_avg'] != 0