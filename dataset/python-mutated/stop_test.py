import streamlit as st
from streamlit.runtime.scriptrunner.script_requests import ScriptRequestType
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StopTest(DeltaGeneratorTestCase):

    def test_stop(self):
        if False:
            i = 10
            return i + 15
        st.stop()
        assert self.script_run_ctx.script_requests._state == ScriptRequestType.STOP