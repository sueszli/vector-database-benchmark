from __future__ import absolute_import
import eventlet
from integration.orquesta import base
from st2common.constants import action as ac_const

class InquiryWiringTest(base.TestWorkflowExecution):

    def test_basic_inquiry(self):
        if False:
            return 10
        ex = self._execute_workflow('examples.orquesta-ask-basic')
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ac_exs = self._wait_for_task(ex, 'get_approval', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(ac_exs[0].id, {'approved': True})
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_consecutive_inquiries(self):
        if False:
            while True:
                i = 10
        ex = self._execute_workflow('examples.orquesta-ask-consecutive')
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_exs = self._wait_for_task(ex, 'get_approval', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(t1_ac_exs[0].id, {'approved': True})
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        t2_ac_exs = self._wait_for_task(ex, 'get_confirmation', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(t2_ac_exs[0].id, {'approved': True})
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_parallel_inquiries(self):
        if False:
            while True:
                i = 10
        ex = self._execute_workflow('examples.orquesta-ask-parallel')
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_exs = self._wait_for_task(ex, 'ask_jack', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(t1_ac_exs[0].id, {'approved': True})
        t1_ac_exs = self._wait_for_task(ex, 'ask_jack', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        eventlet.sleep(2)
        t2_ac_exs = self._wait_for_task(ex, 'ask_jill', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(t2_ac_exs[0].id, {'approved': True})
        t2_ac_exs = self._wait_for_task(ex, 'ask_jill', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_nested_inquiry(self):
        if False:
            i = 10
            return i + 15
        ex = self._execute_workflow('examples.orquesta-ask-nested')
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ac_exs = self._wait_for_task(ex, 'get_approval', ac_const.LIVEACTION_STATUS_PAUSED)
        t2_t2_ac_exs = self._wait_for_task(ac_exs[0], 'get_approval', ac_const.LIVEACTION_STATUS_PENDING)
        self.st2client.inquiries.respond(t2_t2_ac_exs[0].id, {'approved': True})
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)