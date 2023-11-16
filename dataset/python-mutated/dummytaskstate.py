import os
import tempfile
from apps.core.task.coretaskstate import TaskDefinition, Options
from apps.dummy.dummyenvironment import DummyTaskEnvironment
from golem.core.common import get_golem_path
from golem.resource.dirmanager import symlink_or_copy, list_dir_recursive

class DummyTaskDefinition(TaskDefinition):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        TaskDefinition.__init__(self)
        self.options = DummyTaskOptions()
        self.options.difficulty = 4294901760
        self.task_type = 'DUMMY'
        self.shared_data_files = ['in.data']
        self.code_dir = os.path.join(get_golem_path(), 'apps', 'dummy', 'resources', 'code_dir')
        self.code_files = []
        self.result_size = 256
        self.out_file_basename = 'out'
        self.subtasks_count = 5

    def add_to_resources(self):
        if False:
            print('Hello World!')
        super().add_to_resources()
        self.tmp_dir = tempfile.mkdtemp()
        self.shared_data_files = list(self.resources)
        self.code_files = list(list_dir_recursive(self.code_dir))
        symlink_or_copy(self.code_dir, os.path.join(self.tmp_dir, 'code'))
        data_path = os.path.join(self.tmp_dir, 'data')
        data_file = list(self.shared_data_files)[0]
        if os.path.exists(data_path):
            raise FileExistsError('Error adding to resources: data path: {} exists.'.format(data_path))
        os.mkdir(data_path)
        symlink_or_copy(data_file, os.path.join(data_path, os.path.basename(data_file)))
        self.resources = set(list_dir_recursive(self.tmp_dir))

class DummyTaskOptions(Options):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(DummyTaskOptions, self).__init__()
        self.environment = DummyTaskEnvironment()
        self.subtask_data_size = 128
        self.difficulty = 4294901760