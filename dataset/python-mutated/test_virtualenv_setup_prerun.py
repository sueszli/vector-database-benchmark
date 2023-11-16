from st2tests.base import BaseActionTestCase
from pack_mgmt.virtualenv_setup_prerun import PacksTransformationAction

class VirtualenvSetUpPreRunTestCase(BaseActionTestCase):
    action_cls = PacksTransformationAction

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(VirtualenvSetUpPreRunTestCase, self).setUp()

    def test_run_with_pack_list(self):
        if False:
            while True:
                i = 10
        action = self.get_action_instance()
        result = action.run(packs_status={'test1': 'Success.', 'test2': 'Success.'}, packs_list=['test3', 'test4'])
        self.assertEqual(result, ['test3', 'test4', 'test1', 'test2'])

    def test_run_with_none_pack_list(self):
        if False:
            while True:
                i = 10
        action = self.get_action_instance()
        result = action.run(packs_status={'test1': 'Success.', 'test2': 'Success.'}, packs_list=None)
        self.assertEqual(result, ['test1', 'test2'])

    def test_run_with_failed_status(self):
        if False:
            i = 10
            return i + 15
        action = self.get_action_instance()
        result = action.run(packs_status={'test1': 'Failed.', 'test2': 'Success.'}, packs_list=['test3', 'test4'])
        self.assertEqual(result, ['test3', 'test4', 'test2'])