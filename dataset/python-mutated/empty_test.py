import streamlit as st
from streamlit.proto.Empty_pb2 import Empty as EmptyProto
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class StEmmptyAPITest(DeltaGeneratorTestCase):
    """Test Public Streamlit Public APIs."""

    def test_st_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Test st.empty.'
        st.empty()
        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.empty, EmptyProto())