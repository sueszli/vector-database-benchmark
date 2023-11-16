import os
import unittest
import paddle
from paddle import base, static
from paddle.base import core

class Test_XPU_Places(unittest.TestCase):

    def assert_places_equal(self, places0, places1):
        if False:
            print('Hello World!')
        self.assertEqual(len(places0), len(places1))
        for (place0, place1) in zip(places0, places1):
            self.assertEqual(type(place0), type(place1))
            self.assertEqual(place0.get_device_id(), place1.get_device_id())

    def test_check_preset_envs(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_xpu():
            os.environ['FLAGS_selected_xpus'] = '0'
            place_list = static.xpu_places()
            self.assert_places_equal([base.XPUPlace(0)], place_list)

    def test_check_no_preset_envs(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_xpu():
            place_list = static.xpu_places(0)
            self.assert_places_equal([base.XPUPlace(0)], place_list)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()