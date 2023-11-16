from collections import defaultdict
import pytest
from dagster import AssetMaterialization, DagsterInvalidDefinitionError, DagsterTypeCheckDidNotPass, In, Int, List, Nothing, Optional, Out, Output, asset, job, materialize_to_memory, op
from dagster._core.execution.api import create_execution_plan

def _define_nothing_dep_job():
    if False:
        while True:
            i = 10

    @op(out={'complete': Out(Nothing)})
    def start_nothing():
        if False:
            for i in range(10):
                print('nop')
        pass

    @op(ins={'add_complete': In(Nothing), 'yield_complete': In(Nothing)})
    def end_nothing():
        if False:
            i = 10
            return i + 15
        pass

    @op
    def emit_value() -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1

    @op(ins={'on_complete': In(Nothing), 'num': In(Int)})
    def add_value(num) -> int:
        if False:
            print('Hello World!')
        return 1 + num

    @op(name='yield_values', ins={'on_complete': In(Nothing)}, out={'num_1': Out(Int), 'num_2': Out(Int), 'complete': Out(Nothing)})
    def yield_values():
        if False:
            for i in range(10):
                print('nop')
        yield Output(1, 'num_1')
        yield Output(2, 'num_2')
        yield Output(None, 'complete')

    @job
    def simple_exc():
        if False:
            print('Hello World!')
        start_complete = start_nothing()
        (_, _, yield_complete) = yield_values(start_complete)
        end_nothing(add_complete=add_value(on_complete=start_complete, num=emit_value()), yield_complete=yield_complete)
    return simple_exc

def test_valid_nothing_dependencies():
    if False:
        i = 10
        return i + 15
    result = _define_nothing_dep_job().execute_in_process()
    assert result.success

def test_nothing_output_something_input():
    if False:
        while True:
            i = 10

    @op(out=Out(Nothing))
    def do_nothing():
        if False:
            print('Hello World!')
        pass

    @op(ins={'num': In(Int)})
    def add_one(num) -> int:
        if False:
            return 10
        return num + 1

    @job
    def bad_dep():
        if False:
            i = 10
            return i + 15
        add_one(do_nothing())
    with pytest.raises(DagsterTypeCheckDidNotPass):
        bad_dep.execute_in_process()

def test_result_type_check():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(Nothing))
    def bad():
        if False:
            i = 10
            return i + 15
        yield Output('oops')

    @job
    def fail():
        if False:
            print('Hello World!')
        bad()
    with pytest.raises(DagsterTypeCheckDidNotPass):
        fail.execute_in_process()

def test_nothing_inputs():
    if False:
        return 10

    @op(ins={'never_defined': In(Nothing)})
    def emit_one():
        if False:
            while True:
                i = 10
        return 1

    @op
    def emit_two():
        if False:
            for i in range(10):
                print('nop')
        return 2

    @op
    def emit_three():
        if False:
            while True:
                i = 10
        return 3

    @op(out=Out(Nothing))
    def emit_nothing():
        if False:
            print('Hello World!')
        pass

    @op(ins={'_one': In(Nothing), 'one': In(Int), '_two': In(Nothing), 'two': In(Int), '_three': In(Nothing), 'three': In(Int)})
    def adder(one, two, three):
        if False:
            while True:
                i = 10
        assert one == 1
        assert two == 2
        assert three == 3
        return one + two + three

    @job
    def input_test():
        if False:
            print('Hello World!')
        _one = emit_nothing.alias('_one')()
        _two = emit_nothing.alias('_two')()
        _three = emit_nothing.alias('_three')()
        adder(_one=_one, _two=_two, _three=_three, one=emit_one(), two=emit_two(), three=emit_three())
    result = input_test.execute_in_process()
    assert result.success

def test_fanin_deps():
    if False:
        for i in range(10):
            print('nop')
    called = defaultdict(int)

    @op
    def emit_two():
        if False:
            while True:
                i = 10
        return 2

    @op(out=Out(Nothing))
    def emit_nothing():
        if False:
            i = 10
            return i + 15
        called['emit_nothing'] += 1

    @op(ins={'ready': In(Nothing), 'num_1': In(Int), 'num_2': In(Int)})
    def adder(num_1, num_2):
        if False:
            return 10
        assert called['emit_nothing'] == 3
        called['adder'] += 1
        return num_1 + num_2

    @job
    def input_test():
        if False:
            i = 10
            return i + 15
        adder(ready=[emit_nothing.alias('_one')(), emit_nothing.alias('_two')(), emit_nothing.alias('_three')()], num_1=emit_two.alias('emit_1')(), num_2=emit_two.alias('emit_2')())
    result = input_test.execute_in_process()
    assert result.success
    assert called['adder'] == 1
    assert called['emit_nothing'] == 3

def test_valid_nothing_fns():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(Nothing))
    def just_pass():
        if False:
            for i in range(10):
                print('nop')
        pass

    @op(out=Out(Nothing))
    def just_pass2():
        if False:
            i = 10
            return i + 15
        pass

    @op(out=Out(Nothing))
    def ret_none():
        if False:
            i = 10
            return i + 15
        return None

    @op(out=Out(Nothing))
    def yield_none():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None)

    @op(out=Out(Nothing))
    def yield_stuff():
        if False:
            i = 10
            return i + 15
        yield AssetMaterialization.file('/path/to/nowhere')

    @job
    def fn_test():
        if False:
            return 10
        just_pass()
        just_pass2()
        ret_none()
        yield_none()
        yield_stuff()
    result = fn_test.execute_in_process()
    assert result.success
    just_pass()
    just_pass2()
    ret_none()
    [_ for _ in yield_none()]
    [_ for _ in yield_stuff()]

def test_invalid_nothing_fns():
    if False:
        print('Hello World!')

    @op(out=Out(Nothing))
    def ret_val():
        if False:
            while True:
                i = 10
        return 'val'

    @op(out=Out(Nothing))
    def yield_val():
        if False:
            i = 10
            return i + 15
        yield Output('val')
    with pytest.raises(DagsterTypeCheckDidNotPass):

        @job
        def fn_test():
            if False:
                return 10
            ret_val()
        fn_test.execute_in_process()
    with pytest.raises(DagsterTypeCheckDidNotPass):

        @job
        def fn_test2():
            if False:
                return 10
            yield_val()
        fn_test2.execute_in_process()

def test_wrapping_nothing():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError):

        @op(out=Out(List[Nothing]))
        def _():
            if False:
                while True:
                    i = 10
            pass
    with pytest.raises(DagsterInvalidDefinitionError):

        @op(ins={'in': In(List[Nothing])})
        def _(_in):
            if False:
                for i in range(10):
                    print('nop')
            pass
    with pytest.raises(DagsterInvalidDefinitionError):

        @op(out=Out(Optional[Nothing]))
        def _():
            if False:
                for i in range(10):
                    print('nop')
            pass
    with pytest.raises(DagsterInvalidDefinitionError):

        @op(ins={'in': In(Optional[Nothing])})
        def _(_in):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_execution_plan():
    if False:
        i = 10
        return i + 15

    @op(out=Out(Nothing))
    def emit_nothing():
        if False:
            for i in range(10):
                print('nop')
        yield AssetMaterialization.file(path='/path/')

    @op(ins={'ready': In(Nothing)})
    def consume_nothing():
        if False:
            i = 10
            return i + 15
        pass

    @job
    def pipe():
        if False:
            return 10
        consume_nothing(emit_nothing())
    plan = create_execution_plan(pipe)
    levels = plan.get_steps_to_execute_by_level()
    assert 'emit_nothing' in levels[0][0].key
    assert 'consume_nothing' in levels[1][0].key
    assert pipe.execute_in_process().success

def test_nothing_infer():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidDefinitionError, match='which should not be included since no data will be passed for it'):

        @op(ins={'_previous_steps_complete': In(Nothing)})
        def _bad(_previous_steps_complete):
            if False:
                return 10
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match='must be used via In\\(\\) and no parameter should be included in the @op decorated function'):

        @op
        def _bad(_previous_steps_complete: Nothing):
            if False:
                i = 10
                return i + 15
            pass

def test_none_output_non_none_input():
    if False:
        return 10

    @op
    def op1() -> None:
        if False:
            print('Hello World!')
        pass

    @op
    def op2(input1):
        if False:
            while True:
                i = 10
        assert input1 is None

    @job
    def job1():
        if False:
            return 10
        op2(op1())
    assert job1.execute_in_process().success

def test_asset_none_output_non_none_input():
    if False:
        return 10

    @asset
    def asset1() -> None:
        if False:
            i = 10
            return i + 15
        pass

    @asset
    def asset2(asset1):
        if False:
            i = 10
            return i + 15
        assert asset1 is None
    assert materialize_to_memory([asset1, asset2]).success

def test_asset_nothing_output_non_none_input():
    if False:
        print('Hello World!')

    @asset(dagster_type=Nothing)
    def asset1():
        if False:
            return 10
        pass

    @asset
    def asset2(asset1):
        if False:
            print('Hello World!')
        assert asset1 is None
    assert materialize_to_memory([asset1, asset2]).success