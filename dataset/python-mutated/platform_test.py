from parameterized import parameterized
from streamlit.platform import post_parent_message
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class PlatformTest(DeltaGeneratorTestCase):
    """Tests the platform module functions"""

    @parameterized.expand(['Hello', '{"name":"foo", "type":"bar"}'])
    def test_post_parent_message(self, message: str):
        if False:
            while True:
                i = 10
        post_parent_message(message)
        c = self.get_message_from_queue().parent_message
        self.assertEqual(c.message, message)