"""Tests for BzrLibraryState."""
from bzrlib import library_state, tests, ui as _mod_ui
from bzrlib.tests import fixtures

class TestLibraryState(tests.TestCaseWithTransport):

    def test_ui_is_used(self):
        if False:
            for i in range(10):
                print('nop')
        ui = _mod_ui.SilentUIFactory()
        state = library_state.BzrLibraryState(ui=ui, trace=fixtures.RecordingContextManager())
        orig_ui = _mod_ui.ui_factory
        state.__enter__()
        try:
            self.assertEqual(ui, _mod_ui.ui_factory)
        finally:
            state.__exit__(None, None, None)
            self.assertEqual(orig_ui, _mod_ui.ui_factory)

    def test_trace_context(self):
        if False:
            print('Hello World!')
        tracer = fixtures.RecordingContextManager()
        ui = _mod_ui.SilentUIFactory()
        state = library_state.BzrLibraryState(ui=ui, trace=tracer)
        state.__enter__()
        try:
            self.assertEqual(['__enter__'], tracer._calls)
        finally:
            state.__exit__(None, None, None)
            self.assertEqual(['__enter__', '__exit__'], tracer._calls)