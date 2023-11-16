import logging
import os
import random
from copy import copy
from typing import Optional
import enforce
from golem_messages.message import ComputeTaskDef
from apps.core.task.coretask import CoreTask, CoreTaskBuilder, CoreTaskTypeInfo
from apps.dummy.dummyenvironment import DummyTaskEnvironment
from apps.dummy.task.dummytaskstate import DummyTaskOptions
from apps.dummy.task.dummytaskstate import DummyTaskDefinition
from apps.dummy.task.verifier import DummyTaskVerifier
from golem.task.taskbase import Task
from golem.task.taskstate import SubtaskStatus
logger = logging.getLogger('apps.dummy')

class DummyTaskTypeInfo(CoreTaskTypeInfo):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('Dummy', DummyTaskDefinition, DummyTaskOptions, DummyTaskBuilder)

@enforce.runtime_validation(group='dummy')
class DummyTask(CoreTask):
    ENVIRONMENT_CLASS = DummyTaskEnvironment
    VERIFIER_CLASS = DummyTaskVerifier
    RESULT_EXT = '.result'
    TESTING_CHAR = 'a'

    def __init__(self, task_definition: DummyTaskDefinition, root_path=None, owner=None):
        if False:
            print('Hello World!')
        super().__init__(owner=owner, task_definition=task_definition, root_path=root_path)

    def _extra_data(self, perf_index=0.0) -> ComputeTaskDef:
        if False:
            print('Hello World!')
        subtask_id = self.create_subtask_id()
        sbs = self.task_definition.options.subtask_data_size
        data = '{:128x}'.format(random.getrandbits(sbs * 4))
        shared_data_files_base = [os.path.basename(x) for x in self.task_definition.shared_data_files]
        extra_data = {'data_files': shared_data_files_base, 'subtask_data': data, 'difficulty': self.task_definition.options.difficulty, 'result_size': self.task_definition.result_size, 'result_file': self.__get_result_file_name(subtask_id), 'subtask_data_size': sbs, 'entrypoint': 'python3 /golem/scripts/job.py'}
        return self._new_compute_task_def(subtask_id, extra_data, perf_index=perf_index)

    def query_extra_data(self, perf_index: float, node_id: Optional[str]=None, node_name: Optional[str]=None) -> Task.ExtraData:
        if False:
            return 10
        logger.debug('Query extra data on dummytask')
        ctd = self._extra_data(perf_index)
        sid = ctd['subtask_id']
        self.subtasks_given[sid] = copy(ctd['extra_data'])
        self.subtasks_given[sid]['status'] = SubtaskStatus.starting
        self.subtasks_given[sid]['node_id'] = node_id
        self.subtasks_given[sid]['result_extension'] = self.RESULT_EXT
        self.subtasks_given[sid]['shared_data_files'] = self.task_definition.shared_data_files
        self.subtasks_given[sid]['subtask_id'] = sid
        return self.ExtraData(ctd=ctd)

    def accept_results(self, subtask_id, result_files):
        if False:
            print('Hello World!')
        super().accept_results(subtask_id, result_files)
        self.num_tasks_received += 1

    def __get_result_file_name(self, subtask_id: str) -> str:
        if False:
            print('Hello World!')
        return '{}{}{}'.format(self.task_definition.out_file_basename, subtask_id[0:6], self.RESULT_EXT)

    def query_extra_data_for_test_task(self) -> ComputeTaskDef:
        if False:
            print('Hello World!')
        exd = self._extra_data()
        size = self.task_definition.options.subtask_data_size
        char = self.TESTING_CHAR
        exd['extra_data']['subtask_data'] = char * size
        return exd

class DummyTaskBuilder(CoreTaskBuilder):
    TASK_CLASS = DummyTask

    @classmethod
    def build_dictionary(cls, definition: DummyTaskDefinition):
        if False:
            print('Hello World!')
        dictionary = super().build_dictionary(definition)
        opts = dictionary['options']
        opts['subtask_data_size'] = int(definition.options.subtask_data_size)
        opts['difficulty'] = int(definition.options.difficulty)
        return dictionary

    @classmethod
    def build_full_definition(cls, task_type: DummyTaskTypeInfo, dictionary):
        if False:
            print('Hello World!')
        opts = dictionary['options']
        definition = super().build_full_definition(task_type, dictionary)
        sbs = opts.get('subtask_data_size', definition.options.subtask_data_size)
        difficulty = opts.get('difficulty', definition.options.difficulty)
        sbs = int(sbs)
        if isinstance(difficulty, str):
            difficulty = int(difficulty, 16)
        if sbs <= 0:
            raise Exception('Subtask data size should be greater than 0')
        if difficulty < 0:
            raise Exception('Difficulty should be greater than 0')
        definition.options.difficulty = difficulty
        definition.options.subtask_data_size = sbs
        return definition

class DummyTaskMod(DummyTask):

    def query_extra_data(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ctd = self.query_extra_data_for_test_task()
        return self.ExtraData(ctd=ctd)

class DummyTaskBuilderMod(DummyTaskBuilder):
    TASK_CLASS = DummyTaskMod
enforce.config({'groups': {'set': {'dummy': False}}})