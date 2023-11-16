from __future__ import absolute_import
from st2tests.base import BaseActionTestCase
from generate_uuid import GenerateUUID

class GenerateUUIDActionTestCase(BaseActionTestCase):
    action_cls = GenerateUUID

    def test_run(self):
        if False:
            i = 10
            return i + 15
        action = self.get_action_instance()
        result = action.run(uuid_type='uuid1')
        self.assertTrue(result)
        result = action.run(uuid_type='uuid4')
        self.assertTrue(result)
        with self.assertRaises(ValueError):
            result = action.run(uuid_type='foobar')