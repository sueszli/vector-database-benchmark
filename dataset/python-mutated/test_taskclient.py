import unittest
import uuid
from golem.task.taskclient import TaskClient

class TestTaskClient(unittest.TestCase):

    def test_get_or_initialize(self):
        if False:
            for i in range(10):
                print('nop')
        node_id = str(uuid.uuid4())
        node_dict = {}
        tc = TaskClient.get_or_initialize(node_id, node_dict)
        assert tc
        assert node_id in node_dict

    def test_initial_state(self):
        if False:
            i = 10
            return i + 15
        tc = TaskClient()
        assert not tc.should_wait('the hash')
        assert not tc.rejected()

    def test_state_after_start(self):
        if False:
            i = 10
            return i + 15
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=3)
        assert not tc.should_wait('the hash')
        assert not tc.rejected()
        assert tc.should_wait('other hash')

    def test_do_not_allow_to_start_other_WTCT(self):
        if False:
            while True:
                i = 10
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=3)
        assert not tc.start(offer_hash='other hash', num_subtasks=7)

    def test_do_allow_to_start_same_WTCT(self):
        if False:
            i = 10
            return i + 15
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=3)
        assert tc.start(offer_hash='the hash', num_subtasks=3)
        assert not tc.should_wait('the hash')
        assert not tc.rejected()
        assert tc.start(offer_hash='the hash', num_subtasks=3)

    def test_num_subtasks_decrease_allowed_starts(self):
        if False:
            while True:
                i = 10
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=3)
        assert tc.start(offer_hash='the hash', num_subtasks=2)
        assert tc.should_wait('the hash')
        assert not tc.rejected()
        assert not tc.start(offer_hash='the hash', num_subtasks=1)

    def test_do_not_allow_to_start_more_subtasks_than_requested(self):
        if False:
            for i in range(10):
                print('nop')
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=1)
        assert tc.should_wait('the hash')
        assert not tc.rejected()
        assert not tc.start(offer_hash='the hash', num_subtasks=3)

    def test_cancel_allows_more_starts(self):
        if False:
            while True:
                i = 10
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=1)
        tc.cancel()
        assert not tc.should_wait('the hash')
        assert not tc.rejected()
        assert tc.start(offer_hash='the hash', num_subtasks=1)

    def test_not_last_accept_not_resets_state(self):
        if False:
            return 10
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=2)
        assert tc.start(offer_hash='the hash', num_subtasks=2)
        tc.accept()
        assert tc.should_wait('the hash')
        assert tc.should_wait('other hash')
        assert not tc.rejected()
        assert not tc.start(offer_hash='the hash', num_subtasks=2)
        assert not tc.start(offer_hash='other hash', num_subtasks=17)

    def test_last_accept_resets_state(self):
        if False:
            for i in range(10):
                print('nop')
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=1)
        tc.accept()
        assert not tc.should_wait('the hash')
        assert not tc.should_wait('other hash')
        assert not tc.rejected()
        assert tc.start(offer_hash='other hash', num_subtasks=17)

    def test_reject_block_all_starts(self):
        if False:
            print('Hello World!')
        tc = TaskClient()
        assert tc.start(offer_hash='the hash', num_subtasks=1)
        tc.reject()
        assert tc.rejected()
        assert not tc.start(offer_hash='the hash', num_subtasks=1)
        assert not tc.start(offer_hash='other hash', num_subtasks=17)