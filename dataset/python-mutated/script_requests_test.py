import unittest
from typing import Optional
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.scriptrunner.script_requests import RerunData, ScriptRequest, ScriptRequests, ScriptRequestType

def _create_widget(id: str, states: WidgetStates) -> WidgetState:
    if False:
        while True:
            i = 10
    'Create a widget with the given ID.'
    states.widgets.add().id = id
    return states.widgets[-1]

def _get_widget(id: str, states: WidgetStates) -> Optional[WidgetState]:
    if False:
        for i in range(10):
            print('nop')
    'Return the widget with the given ID.'
    for state in states.widgets:
        if state.id == id:
            return state
    return None

class ScriptRequestsTest(unittest.TestCase):

    def test_starts_running(self):
        if False:
            return 10
        'ScriptRequests starts in the CONTINUE state.'
        reqs = ScriptRequests()
        self.assertEqual(ScriptRequestType.CONTINUE, reqs._state)

    def test_stop(self):
        if False:
            return 10
        "A stop request will unconditionally succeed regardless of the\n        ScriptRequests' current state.\n        "
        for state in ScriptRequestType:
            reqs = ScriptRequests()
            reqs._state = state
            reqs.request_stop()
            self.assertEqual(ScriptRequestType.STOP, reqs._state)

    def test_rerun_while_stopped(self):
        if False:
            return 10
        'Requesting a rerun while STOPPED will return False.'
        reqs = ScriptRequests()
        reqs.request_stop()
        success = reqs.request_rerun(RerunData())
        self.assertFalse(success)
        self.assertEqual(ScriptRequestType.STOP, reqs._state)

    def test_rerun_while_running(self):
        if False:
            for i in range(10):
                print('nop')
        'Requesting a rerun while in CONTINUE state will always succeed.'
        reqs = ScriptRequests()
        rerun_data = RerunData(query_string='test_query_string')
        success = reqs.request_rerun(rerun_data)
        self.assertTrue(success)
        self.assertEqual(ScriptRequestType.RERUN, reqs._state)
        self.assertEqual(rerun_data, reqs._rerun_data)

    def test_rerun_coalesce_none_and_none(self):
        if False:
            for i in range(10):
                print('nop')
        'Coalesce two null-WidgetStates rerun requests.'
        reqs = ScriptRequests()
        success = reqs.request_rerun(RerunData(widget_states=None))
        self.assertTrue(success)
        self.assertEqual(ScriptRequestType.RERUN, reqs._state)
        reqs.request_rerun(RerunData(widget_states=None))
        self.assertTrue(success)
        self.assertEqual(ScriptRequestType.RERUN, reqs._state)
        self.assertEqual(RerunData(widget_states=None), reqs._rerun_data)

    def test_rerun_coalesce_widgets_and_widgets(self):
        if False:
            while True:
                i = 10
        'Coalesce two non-null-WidgetStates rerun requests.'
        reqs = ScriptRequests()
        states = WidgetStates()
        _create_widget('trigger', states).trigger_value = True
        _create_widget('int', states).int_value = 123
        success = reqs.request_rerun(RerunData(widget_states=states))
        self.assertTrue(success)
        states = WidgetStates()
        _create_widget('trigger', states).trigger_value = False
        _create_widget('int', states).int_value = 456
        success = reqs.request_rerun(RerunData(widget_states=states))
        self.assertTrue(success)
        self.assertEqual(ScriptRequestType.RERUN, reqs._state)
        result_states = reqs._rerun_data.widget_states
        self.assertEqual(True, _get_widget('trigger', result_states).trigger_value)
        self.assertEqual(456, _get_widget('int', result_states).int_value)

    def test_rerun_coalesce_widgets_and_none(self):
        if False:
            return 10
        'Coalesce a non-null-WidgetStates rerun request with a\n        null-WidgetStates request.\n        '
        reqs = ScriptRequests()
        states = WidgetStates()
        _create_widget('trigger', states).trigger_value = True
        _create_widget('int', states).int_value = 123
        success = reqs.request_rerun(RerunData(widget_states=states))
        self.assertTrue(success)
        success = reqs.request_rerun(RerunData(widget_states=None))
        self.assertTrue(success)
        result_states = reqs._rerun_data.widget_states
        self.assertEqual(True, _get_widget('trigger', result_states).trigger_value)
        self.assertEqual(123, _get_widget('int', result_states).int_value)

    def test_rerun_coalesce_none_and_widgets(self):
        if False:
            return 10
        'Coalesce a null-WidgetStates rerun request with a\n        non-null-WidgetStates request.\n        '
        reqs = ScriptRequests()
        success = reqs.request_rerun(RerunData(widget_states=None))
        self.assertTrue(success)
        states = WidgetStates()
        _create_widget('trigger', states).trigger_value = True
        _create_widget('int', states).int_value = 123
        success = reqs.request_rerun(RerunData(widget_states=states))
        self.assertTrue(success)
        result_states = reqs._rerun_data.widget_states
        self.assertEqual(True, _get_widget('trigger', result_states).trigger_value)
        self.assertEqual(123, _get_widget('int', result_states).int_value)

    def test_on_script_yield_with_no_request(self):
        if False:
            while True:
                i = 10
        'Return None; remain in the CONTINUE state.'
        reqs = ScriptRequests()
        result = reqs.on_scriptrunner_yield()
        self.assertEqual(None, result)
        self.assertEqual(ScriptRequestType.CONTINUE, reqs._state)

    def test_on_script_yield_with_stop_request(self):
        if False:
            for i in range(10):
                print('nop')
        'Return STOP; remain in the STOP state.'
        reqs = ScriptRequests()
        reqs.request_stop()
        result = reqs.on_scriptrunner_yield()
        self.assertEqual(ScriptRequest(ScriptRequestType.STOP), result)
        self.assertEqual(ScriptRequestType.STOP, reqs._state)

    def test_on_script_yield_with_rerun_request(self):
        if False:
            return 10
        'Return RERUN; transition to the CONTINUE state.'
        reqs = ScriptRequests()
        reqs.request_rerun(RerunData())
        result = reqs.on_scriptrunner_yield()
        self.assertEqual(ScriptRequest(ScriptRequestType.RERUN, RerunData()), result)
        self.assertEqual(ScriptRequestType.CONTINUE, reqs._state)

    def test_on_script_complete_with_no_request(self):
        if False:
            return 10
        'Return STOP; transition to the STOP state.'
        reqs = ScriptRequests()
        result = reqs.on_scriptrunner_ready()
        self.assertEqual(ScriptRequest(ScriptRequestType.STOP), result)
        self.assertEqual(ScriptRequestType.STOP, reqs._state)

    def test_on_script_complete_with_pending_request(self):
        if False:
            return 10
        'Return RERUN; transition to the CONTINUE state.'
        reqs = ScriptRequests()
        reqs.request_rerun(RerunData())
        result = reqs.on_scriptrunner_ready()
        self.assertEqual(ScriptRequest(ScriptRequestType.RERUN, RerunData()), result)
        self.assertEqual(ScriptRequestType.CONTINUE, reqs._state)