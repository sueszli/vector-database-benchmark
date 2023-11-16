from helpers import unittest
import luigi

def create_class(cls_name):
    if False:
        return 10

    class NewTask(luigi.WrapperTask):
        pass
    NewTask.__name__ = cls_name
    return NewTask
create_class('MyNewTask')

class SetTaskNameTest(unittest.TestCase):
    """ I accidentally introduced an issue in this commit:
    https://github.com/spotify/luigi/commit/6330e9d0332e6152996292a39c42f752b9288c96

    This causes tasks not to get exposed if they change name later. Adding a unit test
    to resolve the issue. """

    def test_set_task_name(self):
        if False:
            i = 10
            return i + 15
        luigi.run(['--local-scheduler', '--no-lock', 'MyNewTask'])