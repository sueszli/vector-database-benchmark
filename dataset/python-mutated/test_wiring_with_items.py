from __future__ import absolute_import
import os
import tempfile
from integration.orquesta import base
from st2common.constants import action as ac_const

class WithItemsWiringTest(base.TestWorkflowExecution):
    tempfiles = None

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.tempfiles and isinstance(self.tempfiles, list):
            for f in self.tempfiles:
                if os.path.exists(f):
                    os.remove(f)
        self.tempfiles = None
        super(WithItemsWiringTest, self).tearDown()

    def test_with_items(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-with-items'
        members = ['Lakshmi', 'Lindsay', 'Tomaz', 'Matt', 'Drew']
        wf_input = {'members': members}
        message = '%s, resistance is futile!'
        expected_output = {'items': [message % i for i in members]}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_with_items_failure(self):
        if False:
            return 10
        wf_name = 'examples.orquesta-test-with-items-failure'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self._wait_for_task(ex, 'task1', num_task_exs=10)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)

    def test_with_items_concurrency(self):
        if False:
            return 10
        wf_name = 'examples.orquesta-test-with-items'
        concurrency = 2
        num_items = 5
        self.tempfiles = []
        for i in range(0, num_items):
            (_, f) = tempfile.mkstemp()
            os.chmod(f, 493)
            self.tempfiles.append(f)
        wf_input = {'tempfiles': self.tempfiles, 'concurrency': concurrency}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_state(ex, [ac_const.LIVEACTION_STATUS_RUNNING])
        self._wait_for_task(ex, 'task1', num_task_exs=2)
        os.remove(self.tempfiles[0])
        os.remove(self.tempfiles[1])
        self._wait_for_task(ex, 'task1', num_task_exs=4)
        os.remove(self.tempfiles[2])
        os.remove(self.tempfiles[3])
        self._wait_for_task(ex, 'task1', num_task_exs=5)
        os.remove(self.tempfiles[4])
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_with_items_cancellation(self):
        if False:
            return 10
        wf_name = 'examples.orquesta-test-with-items'
        concurrency = 2
        num_items = 2
        self.tempfiles = []
        for i in range(0, num_items):
            (_, f) = tempfile.mkstemp()
            os.chmod(f, 493)
            self.tempfiles.append(f)
        wf_input = {'tempfiles': self.tempfiles, 'concurrency': concurrency}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_state(ex, [ac_const.LIVEACTION_STATUS_RUNNING])
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING, num_task_exs=concurrency)
        self.st2client.executions.delete(ex)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        for f in self.tempfiles:
            os.remove(f)
            self.assertFalse(os.path.exists(f))
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED, num_task_exs=concurrency)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_with_items_concurrency_cancellation(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-with-items'
        concurrency = 2
        num_items = 4
        self.tempfiles = []
        for i in range(0, num_items):
            (_, f) = tempfile.mkstemp()
            os.chmod(f, 493)
            self.tempfiles.append(f)
        wf_input = {'tempfiles': self.tempfiles, 'concurrency': concurrency}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_state(ex, [ac_const.LIVEACTION_STATUS_RUNNING])
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING, num_task_exs=concurrency)
        self.st2client.executions.delete(ex)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        for f in self.tempfiles:
            os.remove(f)
            self.assertFalse(os.path.exists(f))
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED, num_task_exs=concurrency)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_with_items_pause_and_resume(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-with-items'
        num_items = 2
        self.tempfiles = []
        for i in range(0, num_items):
            (_, f) = tempfile.mkstemp()
            os.chmod(f, 493)
            self.tempfiles.append(f)
        wf_input = {'tempfiles': self.tempfiles}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_state(ex, [ac_const.LIVEACTION_STATUS_RUNNING])
        self.st2client.executions.pause(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        for f in self.tempfiles:
            os.remove(f)
            self.assertFalse(os.path.exists(f))
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED, num_task_exs=num_items)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_with_items_concurrency_pause_and_resume(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-test-with-items'
        concurrency = 2
        num_items = 4
        self.tempfiles = []
        for i in range(0, num_items):
            (_, f) = tempfile.mkstemp()
            os.chmod(f, 493)
            self.tempfiles.append(f)
        wf_input = {'tempfiles': self.tempfiles, 'concurrency': concurrency}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_state(ex, [ac_const.LIVEACTION_STATUS_RUNNING])
        self.st2client.executions.pause(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        for f in self.tempfiles[0:concurrency]:
            os.remove(f)
            self.assertFalse(os.path.exists(f))
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED, num_task_exs=concurrency)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        for f in self.tempfiles[concurrency:]:
            os.remove(f)
            self.assertFalse(os.path.exists(f))
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED, num_task_exs=num_items)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_subworkflow_empty_with_items(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-subworkflow-empty-with-items'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)