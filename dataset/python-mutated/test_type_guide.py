import typing
import pytest
import yaml
from dagster import DagsterType, DagsterTypeCheckDidNotPass, In, Nothing, OpExecutionContext, Out, PythonObjectDagsterType, dagster_type_loader, job, make_python_type_usable_as_dagster_type, op, usable_as_dagster_type
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_basic_even_type():
    if False:
        for i in range(10):
            print('nop')
    EvenDagsterType = DagsterType(name='EvenDagsterType', type_check_fn=lambda _, value: isinstance(value, int) and value % 2 == 0)

    @op
    def double_even(num: EvenDagsterType) -> EvenDagsterType:
        if False:
            while True:
                i = 10
        return num
    assert wrap_op_in_graph_and_execute(double_even, input_values={'num': 2}).success
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(double_even, input_values={'num': 3})
    assert not wrap_op_in_graph_and_execute(double_even, input_values={'num': 3}, raise_on_error=False).success

def test_basic_even_type_no_annotations():
    if False:
        i = 10
        return i + 15
    EvenDagsterType = DagsterType(name='EvenDagsterType', type_check_fn=lambda _, value: isinstance(value, int) and value % 2 == 0)

    @op(ins={'num': In(EvenDagsterType)}, out=Out(EvenDagsterType))
    def double_even(num):
        if False:
            for i in range(10):
                print('nop')
        return num
    assert wrap_op_in_graph_and_execute(double_even, input_values={'num': 2}).success
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(double_even, input_values={'num': 3})
    assert not wrap_op_in_graph_and_execute(double_even, input_values={'num': 3}, raise_on_error=False).success

def test_python_object_dagster_type():
    if False:
        i = 10
        return i + 15

    class EvenType:

        def __init__(self, num):
            if False:
                i = 10
                return i + 15
            assert num % 2 == 0
            self.num = num
    EvenDagsterType = PythonObjectDagsterType(EvenType, name='EvenDagsterType')

    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        if False:
            print('Hello World!')
        return EvenType(even_num.num * 2)
    assert wrap_op_in_graph_and_execute(double_even, input_values={'even_num': EvenType(2)}).success
    with pytest.raises(AssertionError):
        wrap_op_in_graph_and_execute(double_even, input_values={'even_num': EvenType(3)})

def test_even_type_loader():
    if False:
        for i in range(10):
            print('nop')

    class EvenType:

        def __init__(self, num):
            if False:
                while True:
                    i = 10
            assert num % 2 == 0
            self.num = num

    @dagster_type_loader(int)
    def load_even_type(_, cfg):
        if False:
            for i in range(10):
                print('nop')
        return EvenType(cfg)
    EvenDagsterType = PythonObjectDagsterType(EvenType, loader=load_even_type)

    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        if False:
            print('Hello World!')
        return EvenType(even_num.num * 2)
    yaml_doc = '\n    ops:\n        double_even:\n            inputs:\n                even_num: 2\n    '
    assert wrap_op_in_graph_and_execute(double_even, run_config=yaml.safe_load(yaml_doc), do_input_mapping=False).success
    assert wrap_op_in_graph_and_execute(double_even, run_config={'ops': {'double_even': {'inputs': {'even_num': 2}}}}, do_input_mapping=False).success
    with pytest.raises(AssertionError):
        wrap_op_in_graph_and_execute(double_even, run_config={'ops': {'double_even': {'inputs': {'even_num': 3}}}}, do_input_mapping=False)

def test_mypy_compliance():
    if False:
        for i in range(10):
            print('nop')

    class EvenType:

        def __init__(self, num):
            if False:
                for i in range(10):
                    print('nop')
            assert num % 2 == 0
            self.num = num
    if typing.TYPE_CHECKING:
        EvenDagsterType = EvenType
    else:
        EvenDagsterType = PythonObjectDagsterType(EvenType)

    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        if False:
            while True:
                i = 10
        return EvenType(even_num.num * 2)
    assert wrap_op_in_graph_and_execute(double_even, input_values={'even_num': EvenType(2)}).success

def test_nothing_type():
    if False:
        print('Hello World!')

    @op(out={'cleanup_done': Out(Nothing)})
    def do_cleanup():
        if False:
            return 10
        pass

    @op(ins={'on_cleanup_done': In(Nothing)})
    def after_cleanup():
        if False:
            while True:
                i = 10
        return 'worked'

    @job
    def nothing_job():
        if False:
            for i in range(10):
                print('nop')
        after_cleanup(do_cleanup())
    result = nothing_job.execute_in_process()
    assert result.success
    assert result.output_for_node('after_cleanup') == 'worked'

def test_nothing_fanin_actually_test():
    if False:
        for i in range(10):
            print('nop')
    ordering = {'counter': 0}

    @op(out=Out(Nothing))
    def start_first_job_section(context: OpExecutionContext):
        if False:
            while True:
                i = 10
        ordering['counter'] += 1
        ordering[context.op.name] = ordering['counter']

    @op(ins={'first_section_done': In(Nothing)}, out=Out(Nothing))
    def perform_clean_up(context: OpExecutionContext):
        if False:
            i = 10
            return i + 15
        ordering['counter'] += 1
        ordering[context.op.name] = ordering['counter']

    @op(ins={'on_cleanup_tasks_done': In(Nothing)})
    def start_next_job_section(context: OpExecutionContext):
        if False:
            print('Hello World!')
        ordering['counter'] += 1
        ordering[context.op.name] = ordering['counter']
        return 'worked'

    @job
    def fanin_job():
        if False:
            return 10
        first_section_done = start_first_job_section()
        start_next_job_section(on_cleanup_tasks_done=[perform_clean_up.alias('cleanup_task_one')(first_section_done), perform_clean_up.alias('cleanup_task_two')(first_section_done)])
    result = fanin_job.execute_in_process()
    assert result.success
    assert ordering['start_first_job_section'] == 1
    assert ordering['start_next_job_section'] == 4

def test_nothing_fanin_empty_body_for_guide():
    if False:
        while True:
            i = 10

    @op(out=Out(Nothing))
    def start_first_job_section():
        if False:
            for i in range(10):
                print('nop')
        pass

    @op(ins={'first_section_done': In(Nothing)}, out=Out(Nothing))
    def perform_clean_up():
        if False:
            i = 10
            return i + 15
        pass

    @op(ins={'on_cleanup_tasks_done': In(Nothing)})
    def start_next_job_section():
        if False:
            while True:
                i = 10
        pass

    @job
    def fanin_job():
        if False:
            return 10
        first_section_done = start_first_job_section()
        start_next_job_section(on_cleanup_tasks_done=[perform_clean_up.alias('cleanup_task_one')(first_section_done), perform_clean_up.alias('cleanup_task_two')(first_section_done)])
    result = fanin_job.execute_in_process()
    assert result.success

def test_usable_as_dagster_type():
    if False:
        i = 10
        return i + 15

    @usable_as_dagster_type
    class EvenType:

        def __init__(self, num):
            if False:
                i = 10
                return i + 15
            assert num % 2 == 0
            self.num = num

    @op
    def double_even(even_num: EvenType) -> EvenType:
        if False:
            while True:
                i = 10
        return EvenType(even_num.num * 2)
    assert wrap_op_in_graph_and_execute(double_even, input_values={'even_num': EvenType(2)}).success

def test_make_usable_as_dagster_type():
    if False:
        while True:
            i = 10

    class EvenType:

        def __init__(self, num):
            if False:
                return 10
            assert num % 2 == 0
            self.num = num
    EvenDagsterType = PythonObjectDagsterType(EvenType, name='EvenDagsterType')
    make_python_type_usable_as_dagster_type(EvenType, EvenDagsterType)

    @op
    def double_even(even_num: EvenType) -> EvenType:
        if False:
            for i in range(10):
                print('nop')
        return EvenType(even_num.num * 2)
    assert wrap_op_in_graph_and_execute(double_even, input_values={'even_num': EvenType(2)}).success