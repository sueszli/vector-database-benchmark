from pyflink.datastream import RuntimeContext
from pyflink.datastream.state import AggregatingStateDescriptor, AggregatingState, ReducingStateDescriptor, ReducingState, MapStateDescriptor, MapState, ListStateDescriptor, ListState, ValueStateDescriptor, ValueState
from pyflink.fn_execution.embedded.state_impl import KeyedStateBackend
from pyflink.fn_execution.metrics.embedded.metric_impl import MetricGroupImpl
from pyflink.metrics import MetricGroup

class StreamingRuntimeContext(RuntimeContext):

    def __init__(self, runtime_context, job_parameters):
        if False:
            i = 10
            return i + 15
        self._runtime_context = runtime_context
        self._job_parameters = job_parameters
        self._keyed_state_backend = None

    def get_task_name(self) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the name of the task in which the UDF runs, as assigned during plan construction.\n        '
        return self._runtime_context.getTaskName()

    def get_number_of_parallel_subtasks(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Gets the parallelism with which the parallel task runs.\n        '
        return self._runtime_context.getNumberOfParallelSubtasks()

    def get_max_number_of_parallel_subtasks(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Gets the number of max-parallelism with which the parallel task runs.\n        '
        return self._runtime_context.getMaxNumberOfParallelSubtasks()

    def get_index_of_this_subtask(self) -> int:
        if False:
            return 10
        '\n        Gets the number of this parallel subtask. The numbering starts from 0 and goes up to\n        parallelism-1 (parallelism as returned by\n        :func:`~RuntimeContext.get_number_of_parallel_subtasks`).\n        '
        return self._runtime_context.getIndexOfThisSubtask()

    def get_attempt_number(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Gets the attempt number of this parallel subtask. First attempt is numbered 0.\n        '
        return self._runtime_context.getAttemptNumber()

    def get_task_name_with_subtasks(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the name of the task, appended with the subtask indicator, such as "MyTask (3/6)",\n        where 3 would be (:func:`~RuntimeContext.get_index_of_this_subtask` + 1), and 6 would be\n        :func:`~RuntimeContext.get_number_of_parallel_subtasks`.\n        '
        return self._runtime_context.getTaskNameWithSubtasks()

    def get_job_parameter(self, key: str, default_value: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the global job parameter value associated with the given key as a string.\n        '
        return self._job_parameters[key] if key in self._job_parameters else default_value

    def get_metrics_group(self) -> MetricGroup:
        if False:
            i = 10
            return i + 15
        return MetricGroupImpl(self._runtime_context.getMetricGroup())

    def get_state(self, state_descriptor: ValueStateDescriptor) -> ValueState:
        if False:
            return 10
        return self._keyed_state_backend.get_value_state(state_descriptor)

    def get_list_state(self, state_descriptor: ListStateDescriptor) -> ListState:
        if False:
            for i in range(10):
                print('nop')
        return self._keyed_state_backend.get_list_state(state_descriptor)

    def get_map_state(self, state_descriptor: MapStateDescriptor) -> MapState:
        if False:
            while True:
                i = 10
        return self._keyed_state_backend.get_map_state(state_descriptor)

    def get_reducing_state(self, state_descriptor: ReducingStateDescriptor) -> ReducingState:
        if False:
            i = 10
            return i + 15
        return self._keyed_state_backend.get_reducing_state(state_descriptor)

    def get_aggregating_state(self, state_descriptor: AggregatingStateDescriptor) -> AggregatingState:
        if False:
            return 10
        return self._keyed_state_backend.get_aggregating_state(state_descriptor)

    def set_keyed_state_backend(self, keyed_state_backend: KeyedStateBackend):
        if False:
            i = 10
            return i + 15
        self._keyed_state_backend = keyed_state_backend

    def get_keyed_state_backend(self):
        if False:
            for i in range(10):
                print('nop')
        return self._keyed_state_backend

    @staticmethod
    def of(runtime_context, job_parameters):
        if False:
            print('Hello World!')
        return StreamingRuntimeContext(runtime_context, job_parameters)