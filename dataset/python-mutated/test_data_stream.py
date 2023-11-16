import datetime
import decimal
import os
import uuid
from collections import defaultdict
from typing import Tuple
from pyflink.common import Row, Configuration
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import WatermarkStrategy, TimestampAssigner
from pyflink.datastream import TimeCharacteristic, RuntimeContext, SlotSharingGroup, StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.data_stream import DataStream
from pyflink.datastream.functions import AggregateFunction, CoMapFunction, CoFlatMapFunction, MapFunction, FilterFunction, FlatMapFunction, KeyedCoProcessFunction, KeyedProcessFunction, KeySelector, ProcessFunction, ReduceFunction, CoProcessFunction, BroadcastProcessFunction, KeyedBroadcastProcessFunction
from pyflink.datastream.output_tag import OutputTag
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor, MapStateDescriptor, ReducingStateDescriptor, ReducingState, AggregatingState, AggregatingStateDescriptor, StateTtlConfig
from pyflink.datastream.tests.test_util import DataStreamTestSinkFunction
from pyflink.java_gateway import get_gateway
from pyflink.metrics import Counter, Meter, Distribution
from pyflink.testing.test_case_utils import PyFlinkBatchTestCase, PyFlinkStreamingTestCase, PyFlinkTestCase
from pyflink.util.java_utils import get_j_env_configuration

class DataStreamTests(object):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(DataStreamTests, self).setUp()
        config = get_j_env_configuration(self.env._j_stream_execution_environment)
        config.setString('pekko.ask.timeout', '20 s')
        self.test_sink = DataStreamTestSinkFunction()

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        self.test_sink.clear()

    def assert_equals_sorted(self, expected, actual):
        if False:
            print('Hello World!')
        expected.sort()
        actual.sort()
        self.assertEqual(expected, actual)

    def test_basic_operations(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.env.from_collection([('ab', Row('a', decimal.Decimal(1))), ('bdc', Row('b', decimal.Decimal(2))), ('cfgs', Row('c', decimal.Decimal(3))), ('deeefg', Row('d', decimal.Decimal(4)))], type_info=Types.TUPLE([Types.STRING(), Types.ROW([Types.STRING(), Types.BIG_DEC()])]))

        class MyMapFunction(MapFunction):

            def map(self, value):
                if False:
                    return 10
                return Row(value[0], value[1] + 1, value[2])

        class MyFlatMapFunction(FlatMapFunction):

            def flat_map(self, value):
                if False:
                    while True:
                        i = 10
                if value[1] % 2 == 0:
                    yield value

        class MyFilterFunction(FilterFunction):

            def filter(self, value):
                if False:
                    return 10
                return value[1] > 2
        ds.map(lambda i: (i[0], len(i[0]), i[1][1]), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.BIG_DEC()])).flat_map(MyFlatMapFunction(), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.BIG_DEC()])).filter(MyFilterFunction()).map(MyMapFunction(), output_type=Types.ROW([Types.STRING(), Types.INT(), Types.BIG_DEC()])).add_sink(self.test_sink)
        self.env.execute('test_basic_operations')
        results = self.test_sink.get_results()
        expected = ['+I[cfgs, 5, 3]', '+I[deeefg, 7, 4]']
        self.assert_equals_sorted(expected, results)

    def test_partition_custom(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2), ('f', 7), ('g', 7), ('h', 8), ('i', 8), ('j', 9)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        expected_num_partitions = 5

        def my_partitioner(key, num_partitions):
            if False:
                return 10
            assert expected_num_partitions == num_partitions
            return key % num_partitions
        partitioned_stream = ds.map(lambda x: x, output_type=Types.ROW([Types.STRING(), Types.INT()])).set_parallelism(4).partition_custom(my_partitioner, lambda x: x[1])
        JPartitionCustomTestMapFunction = get_gateway().jvm.org.apache.flink.python.util.PartitionCustomTestMapFunction
        test_map_stream = DataStream(partitioned_stream._j_data_stream.map(JPartitionCustomTestMapFunction()))
        test_map_stream.set_parallelism(expected_num_partitions).add_sink(self.test_sink)
        self.env.execute('test_partition_custom')

    def test_keyed_process_function_with_state(self):
        if False:
            print('Hello World!')
        self.env.get_config().set_auto_watermark_interval(2000)
        self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
        data_stream = self.env.from_collection([(1, 'hi', '1603708211000'), (2, 'hello', '1603708224000'), (3, 'hi', '1603708226000'), (4, 'hello', '1603708289000'), (5, 'hi', '1603708291000'), (6, 'hello', '1603708293000')], type_info=Types.ROW([Types.INT(), Types.STRING(), Types.STRING()]))

        class MyTimestampAssigner(TimestampAssigner):

            def extract_timestamp(self, value, record_timestamp) -> int:
                if False:
                    i = 10
                    return i + 15
                return int(value[2])

        class MyProcessFunction(KeyedProcessFunction):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.value_state = None
                self.list_state = None
                self.map_state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    return 10
                value_state_descriptor = ValueStateDescriptor('value_state', Types.INT())
                self.value_state = runtime_context.get_state(value_state_descriptor)
                list_state_descriptor = ListStateDescriptor('list_state', Types.INT())
                self.list_state = runtime_context.get_list_state(list_state_descriptor)
                map_state_descriptor = MapStateDescriptor('map_state', Types.INT(), Types.STRING())
                state_ttl_config = StateTtlConfig.new_builder(Time.seconds(1)).set_update_type(StateTtlConfig.UpdateType.OnReadAndWrite).set_state_visibility(StateTtlConfig.StateVisibility.ReturnExpiredIfNotCleanedUp).disable_cleanup_in_background().build()
                map_state_descriptor.enable_time_to_live(state_ttl_config)
                self.map_state = runtime_context.get_map_state(map_state_descriptor)

            def process_element(self, value, ctx):
                if False:
                    for i in range(10):
                        print('nop')
                current_value = self.value_state.value()
                self.value_state.update(value[0])
                current_list = [_ for _ in self.list_state.get()]
                self.list_state.add(value[0])
                map_entries = {k: v for (k, v) in self.map_state.items()}
                keys = sorted(map_entries.keys())
                map_entries_string = [str(k) + ': ' + str(map_entries[k]) for k in keys]
                map_entries_string = '{' + ', '.join(map_entries_string) + '}'
                self.map_state.put(value[0], value[1])
                current_key = ctx.get_current_key()
                yield 'current key: {}, current value state: {}, current list state: {}, current map state: {}, current value: {}'.format(str(current_key), str(current_value), str(current_list), map_entries_string, str(value))

            def on_timer(self, timestamp, ctx):
                if False:
                    i = 10
                    return i + 15
                pass
        watermark_strategy = WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(MyTimestampAssigner())
        data_stream.assign_timestamps_and_watermarks(watermark_strategy).key_by(lambda x: x[1], key_type=Types.STRING()).process(MyProcessFunction(), output_type=Types.STRING()).add_sink(self.test_sink)
        self.env.execute('test time stamp assigner with keyed process function')
        results = self.test_sink.get_results()
        expected = ["current key: hi, current value state: None, current list state: [], current map state: {}, current value: Row(f0=1, f1='hi', f2='1603708211000')", "current key: hello, current value state: None, current list state: [], current map state: {}, current value: Row(f0=2, f1='hello', f2='1603708224000')", "current key: hi, current value state: 1, current list state: [1], current map state: {1: hi}, current value: Row(f0=3, f1='hi', f2='1603708226000')", "current key: hello, current value state: 2, current list state: [2], current map state: {2: hello}, current value: Row(f0=4, f1='hello', f2='1603708289000')", "current key: hi, current value state: 3, current list state: [1, 3], current map state: {1: hi, 3: hi}, current value: Row(f0=5, f1='hi', f2='1603708291000')", "current key: hello, current value state: 4, current list state: [2, 4], current map state: {2: hello, 4: hello}, current value: Row(f0=6, f1='hello', f2='1603708293000')"]
        self.assert_equals_sorted(expected, results)

    def test_reducing_state(self):
        if False:
            print('Hello World!')
        self.env.set_parallelism(2)
        data_stream = self.env.from_collection([(1, 'hi'), (2, 'hello'), (3, 'hi'), (4, 'hello'), (5, 'hi'), (6, 'hello')], type_info=Types.TUPLE([Types.INT(), Types.STRING()]))

        class MyProcessFunction(KeyedProcessFunction):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.reducing_state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    print('Hello World!')
                self.reducing_state = runtime_context.get_reducing_state(ReducingStateDescriptor('reducing_state', lambda i, i2: i + i2, Types.INT()))

            def process_element(self, value, ctx):
                if False:
                    print('Hello World!')
                self.reducing_state.add(value[0])
                yield (self.reducing_state.get(), value[1])
        data_stream.key_by(lambda x: x[1], key_type=Types.STRING()).process(MyProcessFunction(), output_type=Types.TUPLE([Types.INT(), Types.STRING()])).add_sink(self.test_sink)
        self.env.execute('test_reducing_state')
        result = self.test_sink.get_results()
        expected_result = ['(1,hi)', '(2,hello)', '(4,hi)', '(6,hello)', '(9,hi)', '(12,hello)']
        result.sort()
        expected_result.sort()
        self.assertEqual(expected_result, result)

    def test_aggregating_state(self):
        if False:
            while True:
                i = 10
        self.env.set_parallelism(2)
        data_stream = self.env.from_collection([(1, 'hi'), (2, 'hello'), (3, 'hi'), (4, 'hello'), (5, 'hi'), (6, 'hello')], type_info=Types.TUPLE([Types.INT(), Types.STRING()]))

        class MyAggregateFunction(AggregateFunction):

            def create_accumulator(self):
                if False:
                    while True:
                        i = 10
                return 0

            def add(self, value, accumulator):
                if False:
                    i = 10
                    return i + 15
                return value + accumulator

            def get_result(self, accumulator):
                if False:
                    return 10
                return accumulator

            def merge(self, acc_a, acc_b):
                if False:
                    print('Hello World!')
                return acc_a + acc_b

        class MyProcessFunction(KeyedProcessFunction):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.aggregating_state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    while True:
                        i = 10
                descriptor = AggregatingStateDescriptor('aggregating_state', MyAggregateFunction(), Types.INT())
                state_ttl_config = StateTtlConfig.new_builder(Time.seconds(1)).set_update_type(StateTtlConfig.UpdateType.OnReadAndWrite).disable_cleanup_in_background().build()
                descriptor.enable_time_to_live(state_ttl_config)
                self.aggregating_state = runtime_context.get_aggregating_state(descriptor)

            def process_element(self, value, ctx):
                if False:
                    i = 10
                    return i + 15
                self.aggregating_state.add(value[0])
                yield (self.aggregating_state.get(), value[1])
        config = Configuration(j_configuration=get_j_env_configuration(self.env._j_stream_execution_environment))
        config.set_integer('python.fn-execution.bundle.size', 1)
        data_stream.key_by(lambda x: x[1], key_type=Types.STRING()).process(MyProcessFunction(), output_type=Types.TUPLE([Types.INT(), Types.STRING()])).add_sink(self.test_sink)
        self.env.execute('test_aggregating_state')
        results = self.test_sink.get_results()
        expected = ['(1,hi)', '(2,hello)', '(4,hi)', '(6,hello)', '(9,hi)', '(12,hello)']
        self.assert_equals_sorted(expected, results)

    def test_basic_co_operations(self):
        if False:
            i = 10
            return i + 15
        python_file_dir = os.path.join(self.tempdir, 'python_file_dir_' + str(uuid.uuid4()))
        os.mkdir(python_file_dir)
        python_file_path = os.path.join(python_file_dir, 'test_stream_dependency_manage_lib.py')
        with open(python_file_path, 'w') as f:
            f.write('def add_two(a):\n    return a + 2')

        class MyCoFlatMapFunction(CoFlatMapFunction):

            def flat_map1(self, value):
                if False:
                    while True:
                        i = 10
                yield (value + 1)

            def flat_map2(self, value):
                if False:
                    print('Hello World!')
                yield (value - 1)

        class MyCoMapFunction(CoMapFunction):

            def map1(self, value):
                if False:
                    print('Hello World!')
                from test_stream_dependency_manage_lib import add_two
                return add_two(value)

            def map2(self, value):
                if False:
                    i = 10
                    return i + 15
                return value + 1
        self.env.add_python_file(python_file_path)
        ds_1 = self.env.from_collection([1, 2, 3, 4, 5])
        ds_2 = ds_1.map(lambda x: x * 2)
        ds_1.connect(ds_2).flat_map(MyCoFlatMapFunction()).connect(ds_2).map(MyCoMapFunction()).add_sink(self.test_sink)
        self.env.execute('test_basic_co_operations')
        results = self.test_sink.get_results(True)
        expected = ['4', '5', '6', '7', '8', '3', '5', '7', '9', '11', '3', '5', '7', '9', '11']
        self.assert_equals_sorted(expected, results)

    def test_keyed_co_process(self):
        if False:
            return 10
        self.env.set_parallelism(1)
        ds1 = self.env.from_collection([('a', 1), ('b', 2), ('c', 3)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds2 = self.env.from_collection([('b', 2), ('c', 3), ('d', 4)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds1 = ds1.assign_timestamps_and_watermarks(WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(SecondColumnTimestampAssigner()))
        ds2 = ds2.assign_timestamps_and_watermarks(WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(SecondColumnTimestampAssigner()))
        ds1.connect(ds2).key_by(lambda x: x[0], lambda x: x[0]).process(MyKeyedCoProcessFunction()).map(lambda x: Row(x[0], x[1] + 1)).add_sink(self.test_sink)
        self.env.execute('test_keyed_co_process_function')
        results = self.test_sink.get_results(True)
        expected = ["<Row('a', 2)>", "<Row('b', 2)>", "<Row('b', 3)>", "<Row('c', 2)>", "<Row('c', 3)>", "<Row('d', 2)>", "<Row('on_timer', 4)>"]
        self.assert_equals_sorted(expected, results)

    def test_co_broadcast_process(self):
        if False:
            i = 10
            return i + 15
        ds = self.env.from_collection([1, 2, 3, 4, 5], type_info=Types.INT())
        ds_broadcast = self.env.from_collection([(0, 'a'), (1, 'b')], type_info=Types.TUPLE([Types.INT(), Types.STRING()]))

        class MyBroadcastProcessFunction(BroadcastProcessFunction):

            def __init__(self, map_state_desc):
                if False:
                    return 10
                self._map_state_desc = map_state_desc
                self._cache = defaultdict(list)

            def process_element(self, value: int, ctx: BroadcastProcessFunction.ReadOnlyContext):
                if False:
                    return 10
                ro_broadcast_state = ctx.get_broadcast_state(self._map_state_desc)
                key = value % 2
                if ro_broadcast_state.contains(key):
                    if self._cache.get(key) is not None:
                        for v in self._cache[key]:
                            yield (ro_broadcast_state.get(key) + str(v))
                        self._cache[key].clear()
                    yield (ro_broadcast_state.get(key) + str(value))
                else:
                    self._cache[key].append(value)

            def process_broadcast_element(self, value: Tuple[int, str], ctx: BroadcastProcessFunction.Context):
                if False:
                    while True:
                        i = 10
                key = value[0]
                yield (str(key) + value[1])
                broadcast_state = ctx.get_broadcast_state(self._map_state_desc)
                broadcast_state.put(key, value[1])
                if self._cache.get(key) is not None:
                    for v in self._cache[key]:
                        yield (value[1] + str(v))
                    self._cache[key].clear()
        map_state_desc = MapStateDescriptor('mapping', key_type_info=Types.INT(), value_type_info=Types.STRING())
        ds.connect(ds_broadcast.broadcast(map_state_desc)).process(MyBroadcastProcessFunction(map_state_desc), output_type=Types.STRING()).add_sink(self.test_sink)
        self.env.execute('test_co_broadcast_process')
        expected = ['0a', '0a', '1b', '1b', 'a2', 'a4', 'b1', 'b3', 'b5']
        self.assert_equals_sorted(expected, self.test_sink.get_results())

    def test_keyed_co_broadcast_process(self):
        if False:
            i = 10
            return i + 15
        ds = self.env.from_collection([(1, '1603708211000'), (2, '1603708212000'), (3, '1603708213000'), (4, '1603708214000')], type_info=Types.ROW([Types.INT(), Types.STRING()]))
        ds_broadcast = self.env.from_collection([(0, '1603708215000', 'a'), (1, '1603708215000', 'b')], type_info=Types.ROW([Types.INT(), Types.STRING(), Types.STRING()]))
        watermark_strategy = WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(SecondColumnTimestampAssigner())
        ds = ds.assign_timestamps_and_watermarks(watermark_strategy)
        ds_broadcast = ds_broadcast.assign_timestamps_and_watermarks(watermark_strategy)

        def _create_string(s, t):
            if False:
                while True:
                    i = 10
            return 'value: {}, ts: {}'.format(s, t)

        class MyKeyedBroadcastProcessFunction(KeyedBroadcastProcessFunction):

            def __init__(self, map_state_desc):
                if False:
                    i = 10
                    return i + 15
                self._map_state_desc = map_state_desc
                self._cache = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    return 10
                self._cache = defaultdict(list)

            def process_element(self, value: Tuple[int, str], ctx: KeyedBroadcastProcessFunction.ReadOnlyContext):
                if False:
                    return 10
                ro_broadcast_state = ctx.get_broadcast_state(self._map_state_desc)
                key = value[0] % 2
                if ro_broadcast_state.contains(key):
                    if self._cache.get(key) is not None:
                        for v in self._cache[key]:
                            yield _create_string(ro_broadcast_state.get(key) + str(v[0]), v[1])
                        self._cache[key].clear()
                    yield _create_string(ro_broadcast_state.get(key) + str(value[0]), value[1])
                else:
                    self._cache[key].append(value)
                ctx.timer_service().register_event_time_timer(ctx.timestamp() + 10000)

            def process_broadcast_element(self, value: Tuple[int, str, str], ctx: KeyedBroadcastProcessFunction.Context):
                if False:
                    while True:
                        i = 10
                key = value[0]
                yield _create_string(str(key) + value[2], ctx.timestamp())
                broadcast_state = ctx.get_broadcast_state(self._map_state_desc)
                broadcast_state.put(key, value[2])
                if self._cache.get(key) is not None:
                    for v in self._cache[key]:
                        yield _create_string(value[2] + str(v[0]), v[1])
                    self._cache[key].clear()

            def on_timer(self, timestamp: int, ctx: KeyedBroadcastProcessFunction.OnTimerContext):
                if False:
                    return 10
                yield _create_string(ctx.get_current_key(), timestamp)
        map_state_desc = MapStateDescriptor('mapping', key_type_info=Types.INT(), value_type_info=Types.STRING())
        ds.key_by(lambda t: t[0]).connect(ds_broadcast.broadcast(map_state_desc)).process(MyKeyedBroadcastProcessFunction(map_state_desc), output_type=Types.STRING()).add_sink(self.test_sink)
        self.env.execute('test_keyed_co_broadcast_process')
        expected = ['value: 0a, ts: 1603708215000', 'value: 0a, ts: 1603708215000', 'value: 1, ts: 1603708221000', 'value: 1b, ts: 1603708215000', 'value: 1b, ts: 1603708215000', 'value: 2, ts: 1603708222000', 'value: 3, ts: 1603708223000', 'value: 4, ts: 1603708224000', 'value: a2, ts: 1603708212000', 'value: a4, ts: 1603708214000', 'value: b1, ts: 1603708211000', 'value: b3, ts: 1603708213000']
        self.assert_equals_sorted(expected, self.test_sink.get_results())

    def test_process_side_output(self):
        if False:
            print('Hello World!')
        tag = OutputTag('side', Types.INT())
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    return 10
                yield value[0]
                yield (tag, value[1])
        ds2 = ds.process(MyProcessFunction(), output_type=Types.STRING())
        main_sink = DataStreamTestSinkFunction()
        ds2.add_sink(main_sink)
        side_sink = DataStreamTestSinkFunction()
        ds2.get_side_output(tag).add_sink(side_sink)
        self.env.execute('test_process_side_output')
        main_expected = ['a', 'b', 'c']
        self.assert_equals_sorted(main_expected, main_sink.get_results())
        side_expected = ['0', '1', '2']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_side_output_chained_with_upstream_operator(self):
        if False:
            i = 10
            return i + 15
        tag = OutputTag('side', Types.INT())
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    for i in range(10):
                        print('nop')
                yield value[0]
                yield (tag, value[1])
        ds2 = ds.map(lambda e: (e[0], e[1] + 1)).process(MyProcessFunction(), output_type=Types.STRING())
        main_sink = DataStreamTestSinkFunction()
        ds2.add_sink(main_sink)
        side_sink = DataStreamTestSinkFunction()
        ds2.get_side_output(tag).add_sink(side_sink)
        self.env.execute('test_side_output_chained_with_upstream_operator')
        main_expected = ['a', 'b', 'c']
        self.assert_equals_sorted(main_expected, main_sink.get_results())
        side_expected = ['1', '2', '3']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_process_multiple_side_output(self):
        if False:
            for i in range(10):
                print('nop')
        tag1 = OutputTag('side1', Types.INT())
        tag2 = OutputTag('side2', Types.STRING())
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    i = 10
                    return i + 15
                yield value[0]
                yield (tag1, value[1])
                yield (tag2, value[0] + str(value[1]))
        ds2 = ds.process(MyProcessFunction(), output_type=Types.STRING())
        main_sink = DataStreamTestSinkFunction()
        ds2.add_sink(main_sink)
        side1_sink = DataStreamTestSinkFunction()
        ds2.get_side_output(tag1).add_sink(side1_sink)
        side2_sink = DataStreamTestSinkFunction()
        ds2.get_side_output(tag2).add_sink(side2_sink)
        self.env.execute('test_process_multiple_side_output')
        main_expected = ['a', 'b', 'c']
        self.assert_equals_sorted(main_expected, main_sink.get_results())
        side1_expected = ['0', '1', '2']
        self.assert_equals_sorted(side1_expected, side1_sink.get_results())
        side2_expected = ['a0', 'b1', 'c2']
        self.assert_equals_sorted(side2_expected, side2_sink.get_results())

    def test_co_process_side_output(self):
        if False:
            print('Hello World!')
        tag = OutputTag('side', Types.INT())

        class MyCoProcessFunction(CoProcessFunction):

            def process_element1(self, value, ctx: 'CoProcessFunction.Context'):
                if False:
                    i = 10
                    return i + 15
                yield value[0]
                yield (tag, value[1])

            def process_element2(self, value, ctx: 'CoProcessFunction.Context'):
                if False:
                    print('Hello World!')
                yield value[1]
                yield (tag, value[0])
        ds1 = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds2 = self.env.from_collection([(3, 'c'), (1, 'a'), (0, 'd')], type_info=Types.ROW([Types.INT(), Types.STRING()]))
        ds3 = ds1.connect(ds2).process(MyCoProcessFunction(), output_type=Types.STRING())
        ds3.add_sink(self.test_sink)
        side_sink = DataStreamTestSinkFunction()
        ds3.get_side_output(tag).add_sink(side_sink)
        self.env.execute('test_co_process_side_output')
        main_expected = ['a', 'a', 'b', 'c', 'c', 'd']
        self.assert_equals_sorted(main_expected, self.test_sink.get_results())
        side_expected = ['0', '0', '1', '1', '2', '3']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_co_broadcast_side_output(self):
        if False:
            i = 10
            return i + 15
        tag = OutputTag('side', Types.INT())

        class MyBroadcastProcessFunction(BroadcastProcessFunction):

            def process_element(self, value, ctx):
                if False:
                    return 10
                yield value[0]
                yield (tag, value[1])

            def process_broadcast_element(self, value, ctx):
                if False:
                    for i in range(10):
                        print('nop')
                yield value[1]
                yield (tag, value[0])
        self.env.set_parallelism(2)
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds_broadcast = self.env.from_collection([(3, 'd'), (4, 'f')], type_info=Types.ROW([Types.INT(), Types.STRING()]))
        map_state_desc = MapStateDescriptor('dummy', key_type_info=Types.INT(), value_type_info=Types.STRING())
        ds = ds.connect(ds_broadcast.broadcast(map_state_desc)).process(MyBroadcastProcessFunction(), output_type=Types.STRING())
        side_sink = DataStreamTestSinkFunction()
        ds.get_side_output(tag).add_sink(side_sink)
        ds.add_sink(self.test_sink)
        self.env.execute('test_co_broadcast_process_side_output')
        main_expected = ['a', 'b', 'c', 'd', 'd', 'f', 'f']
        self.assert_equals_sorted(main_expected, self.test_sink.get_results())
        side_expected = ['0', '1', '2', '3', '3', '4', '4']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_keyed_process_side_output(self):
        if False:
            while True:
                i = 10
        tag = OutputTag('side', Types.INT())
        ds = self.env.from_collection([('a', 1), ('b', 2), ('a', 3), ('b', 4)], type_info=Types.ROW([Types.STRING(), Types.INT()]))

        class MyKeyedProcessFunction(KeyedProcessFunction):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.reducing_state = None

            def open(self, context: RuntimeContext):
                if False:
                    i = 10
                    return i + 15
                self.reducing_state = context.get_reducing_state(ReducingStateDescriptor('reduce', lambda i, j: i + j, Types.INT()))

            def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
                if False:
                    while True:
                        i = 10
                yield value[1]
                self.reducing_state.add(value[1])
                yield (tag, self.reducing_state.get())
        ds2 = ds.key_by(lambda e: e[0]).process(MyKeyedProcessFunction(), output_type=Types.INT())
        main_sink = DataStreamTestSinkFunction()
        ds2.add_sink(main_sink)
        side_sink = DataStreamTestSinkFunction()
        ds2.get_side_output(tag).add_sink(side_sink)
        self.env.execute('test_keyed_process_side_output')
        main_expected = ['1', '2', '3', '4']
        self.assert_equals_sorted(main_expected, main_sink.get_results())
        side_expected = ['1', '2', '4', '6']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_keyed_co_process_side_output(self):
        if False:
            return 10
        tag = OutputTag('side', Types.INT())
        ds1 = self.env.from_collection([('a', 1), ('b', 2), ('a', 3), ('b', 4)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds2 = self.env.from_collection([(8, 'a'), (7, 'b'), (6, 'a'), (5, 'b')], type_info=Types.ROW([Types.INT(), Types.STRING()]))

        class MyKeyedCoProcessFunction(KeyedCoProcessFunction):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.reducing_state = None

            def open(self, context: RuntimeContext):
                if False:
                    return 10
                self.reducing_state = context.get_reducing_state(ReducingStateDescriptor('reduce', lambda i, j: i + j, Types.INT()))

            def process_element1(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                if False:
                    for i in range(10):
                        print('nop')
                yield (ctx.get_current_key(), value[1])
                self.reducing_state.add(1)
                yield (tag, self.reducing_state.get())

            def process_element2(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                if False:
                    while True:
                        i = 10
                yield (ctx.get_current_key(), value[0])
                self.reducing_state.add(1)
                yield (tag, self.reducing_state.get())
        ds3 = ds1.key_by(lambda e: e[0]).connect(ds2.key_by(lambda e: e[1])).process(MyKeyedCoProcessFunction(), output_type=Types.TUPLE([Types.STRING(), Types.INT()]))
        main_sink = DataStreamTestSinkFunction()
        ds3.add_sink(main_sink)
        side_sink = DataStreamTestSinkFunction()
        ds3.get_side_output(tag).add_sink(side_sink)
        self.env.execute('test_keyed_co_process_side_output')
        main_expected = ['(a,1)', '(b,2)', '(a,3)', '(b,4)', '(b,5)', '(a,6)', '(b,7)', '(a,8)']
        self.assert_equals_sorted(main_expected, main_sink.get_results())
        side_expected = ['1', '1', '2', '2', '3', '3', '4', '4']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_keyed_co_broadcast_side_output(self):
        if False:
            i = 10
            return i + 15
        tag = OutputTag('side', Types.INT())

        class MyKeyedBroadcastProcessFunction(KeyedBroadcastProcessFunction):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.reducing_state = None

            def open(self, context: RuntimeContext):
                if False:
                    return 10
                self.reducing_state = context.get_reducing_state(ReducingStateDescriptor('reduce', lambda i, j: i + j, Types.INT()))

            def process_element(self, value, ctx):
                if False:
                    i = 10
                    return i + 15
                self.reducing_state.add(value[1])
                yield value[0]
                yield (tag, self.reducing_state.get())

            def process_broadcast_element(self, value, ctx):
                if False:
                    while True:
                        i = 10
                yield value[1]
                yield (tag, value[0])
        self.env.set_parallelism(2)
        ds = self.env.from_collection([('a', 0), ('b', 1), ('a', 2), ('b', 3)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds_broadcast = self.env.from_collection([(5, 'c'), (6, 'd')], type_info=Types.ROW([Types.INT(), Types.STRING()]))
        map_state_desc = MapStateDescriptor('dummy', key_type_info=Types.INT(), value_type_info=Types.STRING())
        ds = ds.key_by(lambda e: e[0]).connect(ds_broadcast.broadcast(map_state_desc)).process(MyKeyedBroadcastProcessFunction(), output_type=Types.STRING())
        side_sink = DataStreamTestSinkFunction()
        ds.get_side_output(tag).add_sink(side_sink)
        ds.add_sink(self.test_sink)
        self.env.execute('test_keyed_co_broadcast_process_side_output')
        main_expected = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
        self.assert_equals_sorted(main_expected, self.test_sink.get_results())
        side_expected = ['0', '1', '2', '4', '5', '5', '6', '6']
        self.assert_equals_sorted(side_expected, side_sink.get_results())

    def test_side_output_stream_execute_and_collect(self):
        if False:
            print('Hello World!')
        tag = OutputTag('side', Types.INT())

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx):
                if False:
                    print('Hello World!')
                yield value
                yield (tag, value * 2)
        ds = self.env.from_collection([1, 2, 3], Types.INT()).process(MyProcessFunction())
        ds_side = ds.get_side_output(tag)
        result = [i for i in ds_side.execute_and_collect()]
        expected = [2, 4, 6]
        self.assert_equals_sorted(expected, result)

    def test_side_output_tag_reusing(self):
        if False:
            for i in range(10):
                print('nop')
        tag = OutputTag('side', Types.INT())

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx):
                if False:
                    print('Hello World!')
                yield value
                yield (tag, value * 2)
        side1_sink = DataStreamTestSinkFunction()
        ds = self.env.from_collection([1, 2, 3], Types.INT()).process(MyProcessFunction())
        ds.get_side_output(tag).add_sink(side1_sink)
        side2_sink = DataStreamTestSinkFunction()
        ds.map(lambda i: i * 2).process(MyProcessFunction()).get_side_output(tag).add_sink(side2_sink)
        self.env.execute('test_side_output_tag_reusing')
        result1 = [i for i in side1_sink.get_results(stringify=False)]
        result2 = [i for i in side2_sink.get_results(stringify=False)]
        self.assert_equals_sorted(['2', '4', '6'], result1)
        self.assert_equals_sorted(['4', '8', '12'], result2)

class DataStreamStreamingTests(DataStreamTests):

    def test_reduce_with_state(self):
        if False:
            print('Hello World!')
        ds = self.env.from_collection([('a', 0), ('c', 1), ('d', 1), ('b', 0), ('e', 1)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector(), key_type=Types.INT())
        with self.assertRaises(Exception):
            keyed_stream.name('keyed stream')
        keyed_stream.reduce(MyReduceFunction()).add_sink(self.test_sink)
        self.env.execute('key_by_test')
        results = self.test_sink.get_results(False)
        expected = ['+I[a, 0]', '+I[ab, 0]', '+I[c, 1]', '+I[cd, 1]', '+I[cde, 1]']
        self.assert_equals_sorted(expected, results)

class DataStreamBatchTests(DataStreamTests):

    def test_reduce_with_state(self):
        if False:
            return 10
        ds = self.env.from_collection([('a', 0), ('c', 1), ('d', 1), ('b', 0), ('e', 1)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector(), key_type=Types.INT())
        with self.assertRaises(Exception):
            keyed_stream.name('keyed stream')
        keyed_stream.reduce(MyReduceFunction()).add_sink(self.test_sink)
        self.env.execute('key_by_test')
        results = self.test_sink.get_results(False)
        expected = ['+I[ab, 0]', '+I[cde, 1]']
        self.assert_equals_sorted(expected, results)

class ProcessDataStreamTests(DataStreamTests):
    """
    The tests only tested in Process Mode.
    """

    def test_basic_co_operations_with_output_type(self):
        if False:
            print('Hello World!')

        class MyCoMapFunction(CoMapFunction):

            def map1(self, value):
                if False:
                    print('Hello World!')
                return value + 2

            def map2(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                return value + 1

        class MyCoFlatMapFunction(CoFlatMapFunction):

            def flat_map1(self, value):
                if False:
                    print('Hello World!')
                yield (value + 1)

            def flat_map2(self, value):
                if False:
                    return 10
                yield (value - 1)
        ds_1 = self.env.from_collection([1, 2, 3, 4, 5])
        ds_2 = ds_1.map(lambda x: x * 2)
        ds_1.connect(ds_2).flat_map(MyCoFlatMapFunction(), output_type=Types.INT()).connect(ds_2).map(MyCoMapFunction(), output_type=Types.INT()).add_sink(self.test_sink)
        self.env.execute('test_basic_co_operations_with_output_type')
        results = self.test_sink.get_results()
        expected = ['4', '5', '6', '7', '8', '3', '5', '7', '9', '11', '3', '5', '7', '9', '11']
        self.assert_equals_sorted(expected, results)

    def test_keyed_co_map(self):
        if False:
            for i in range(10):
                print('nop')
        ds1 = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()])).key_by(MyKeySelector(), key_type=Types.INT())
        ds2 = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))

        class AssertKeyCoMapFunction(CoMapFunction):

            def __init__(self):
                if False:
                    return 10
                self.pre1 = None
                self.pre2 = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    print('Hello World!')
                self.pre1 = runtime_context.get_state(ValueStateDescriptor('pre1', Types.STRING()))
                self.pre2 = runtime_context.get_state(ValueStateDescriptor('pre2', Types.STRING()))

            def map1(self, value):
                if False:
                    return 10
                if value[0] == 'b':
                    assert self.pre1.value() == 'a'
                if value[0] == 'd':
                    assert self.pre1.value() == 'c'
                self.pre1.update(value[0])
                return value

            def map2(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                if value[0] == 'b':
                    assert self.pre2.value() == 'a'
                if value[0] == 'd':
                    assert self.pre2.value() == 'c'
                self.pre2.update(value[0])
                return value
        ds1.connect(ds2).key_by(MyKeySelector(), MyKeySelector(), key_type=Types.INT()).map(AssertKeyCoMapFunction()).map(lambda x: (x[0], x[1] + 1)).add_sink(self.test_sink)
        self.env.execute()
        results = self.test_sink.get_results(True)
        expected = ["('e', 3)", "('a', 1)", "('b', 1)", "('c', 2)", "('d', 2)", "('e', 3)", "('a', 1)", "('b', 1)", "('c', 2)", "('d', 2)"]
        self.assert_equals_sorted(expected, results)

    def test_keyed_co_flat_map(self):
        if False:
            i = 10
            return i + 15
        ds1 = self.env.from_collection([(1, 1), (2, 2), (3, 3)], type_info=Types.ROW([Types.INT(), Types.INT()]))
        ds2 = self.env.from_collection([('a', 'a'), ('b', 'b'), ('c', 'c'), ('a', 'a')], type_info=Types.ROW([Types.STRING(), Types.STRING()]))
        ds1.connect(ds2).key_by(lambda x: 1, lambda x: 1).flat_map(MyRichCoFlatMapFunction(), output_type=Types.STRING()).filter(lambda x: x != '4').add_sink(self.test_sink)
        self.env.execute('test_keyed_co_flat_map')
        results = self.test_sink.get_results(False)
        expected = ['2', '2', '3', '3', 'a', 'b', 'c']
        self.assert_equals_sorted(expected, results)

    def test_keyed_map(self):
        if False:
            for i in range(10):
                print('nop')
        from pyflink.util.java_utils import get_j_env_configuration
        from pyflink.common import Configuration
        config = Configuration(j_configuration=get_j_env_configuration(self.env._j_stream_execution_environment))
        config.set_integer('python.fn-execution.bundle.size', 1)
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 0), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector(), key_type=Types.INT())
        with self.assertRaises(Exception):
            keyed_stream.name('keyed stream')

        class AssertKeyMapFunction(MapFunction):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    return 10
                self.state = runtime_context.get_state(ValueStateDescriptor('test_state', Types.INT()))

            def map(self, value):
                if False:
                    while True:
                        i = 10
                if value[0] == 'a':
                    pass
                elif value[0] == 'b':
                    state_value = self._get_state_value()
                    assert state_value == 1
                    self.state.update(state_value)
                elif value[0] == 'c':
                    state_value = self._get_state_value()
                    assert state_value == 1
                    self.state.update(state_value)
                elif value[0] == 'd':
                    state_value = self._get_state_value()
                    assert state_value == 2
                    self.state.update(state_value)
                else:
                    pass
                return value

            def _get_state_value(self):
                if False:
                    print('Hello World!')
                state_value = self.state.value()
                if state_value is None:
                    state_value = 1
                else:
                    state_value += 1
                return state_value
        keyed_stream.map(AssertKeyMapFunction()).map(lambda x: (x[0], x[1] + 1)).add_sink(self.test_sink)
        self.env.execute('test_keyed_map')
        results = self.test_sink.get_results(True)
        expected = ["('e', 3)", "('a', 1)", "('b', 2)", "('c', 1)", "('d', 2)"]
        self.assert_equals_sorted(expected, results)

    def test_keyed_flat_map(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector(), key_type=Types.INT())
        with self.assertRaises(Exception):
            keyed_stream.name('keyed stream')

        class AssertKeyMapFunction(FlatMapFunction):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.pre = None
                self.state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    i = 10
                    return i + 15
                self.state = runtime_context.get_state(ValueStateDescriptor('test_state', Types.INT()))

            def flat_map(self, value):
                if False:
                    return 10
                state_value = self.state.value()
                if state_value is None:
                    state_value = 1
                else:
                    state_value += 1
                if value[0] == 'b':
                    assert self.pre == 'a'
                    assert state_value == 2
                if value[0] == 'd':
                    assert self.pre == 'c'
                    assert state_value == 2
                if value[0] == 'e':
                    assert state_value == 1
                self.pre = value[0]
                self.state.update(state_value)
                yield value
        keyed_stream.flat_map(AssertKeyMapFunction()).map(lambda x: (x[0], x[1] + 1)).add_sink(self.test_sink)
        self.env.execute('test_keyed_flat_map')
        results = self.test_sink.get_results(True)
        expected = ["('e', 3)", "('a', 1)", "('b', 1)", "('c', 2)", "('d', 2)"]
        self.assert_equals_sorted(expected, results)

    def test_keyed_filter(self):
        if False:
            while True:
                i = 10
        ds = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector())
        with self.assertRaises(Exception):
            keyed_stream.name('keyed stream')

        class AssertKeyFilterFunction(FilterFunction):

            def __init__(self):
                if False:
                    return 10
                self.pre = None
                self.state = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    while True:
                        i = 10
                self.state = runtime_context.get_state(ValueStateDescriptor('test_state', Types.INT()))

            def filter(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                state_value = self.state.value()
                if state_value is None:
                    state_value = 1
                else:
                    state_value += 1
                if value[0] == 'b':
                    assert self.pre == 'a'
                    assert state_value == 2
                    return False
                if value[0] == 'd':
                    assert self.pre == 'c'
                    assert state_value == 2
                    return False
                if value[0] == 'e':
                    assert state_value == 1
                self.pre = value[0]
                self.state.update(state_value)
                return True
        keyed_stream.filter(AssertKeyFilterFunction()).filter(lambda x: x[1] > 0).add_sink(self.test_sink)
        self.env.execute('key_by_test')
        results = self.test_sink.get_results(False)
        expected = ['+I[c, 1]', '+I[e, 2]']
        self.assert_equals_sorted(expected, results)

    def test_multi_key_by(self):
        if False:
            return 10
        ds = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds.key_by(MyKeySelector(), key_type=Types.INT()).key_by(lambda x: x[0]).add_sink(self.test_sink)
        self.env.execute('test multi key by')
        results = self.test_sink.get_results(False)
        expected = ['+I[d, 1]', '+I[c, 1]', '+I[a, 0]', '+I[b, 0]', '+I[e, 2]']
        self.assert_equals_sorted(expected, results)

    def test_collection_type_info(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.env.from_collection([(1, [1.1, 1.2, 1.3], [None, 'hi', 'flink'], datetime.date(2021, 1, 9), datetime.time(12, 0, 0), datetime.datetime(2021, 1, 9, 12, 0, 0, 11000), [1, 2, 3])], type_info=Types.ROW([Types.INT(), Types.PRIMITIVE_ARRAY(Types.FLOAT()), Types.BASIC_ARRAY(Types.STRING()), Types.SQL_DATE(), Types.SQL_TIME(), Types.SQL_TIMESTAMP(), Types.LIST(Types.INT())]))
        ds.map(lambda x: x, output_type=Types.ROW([Types.INT(), Types.PRIMITIVE_ARRAY(Types.FLOAT()), Types.BASIC_ARRAY(Types.STRING()), Types.SQL_DATE(), Types.SQL_TIME(), Types.SQL_TIMESTAMP(), Types.LIST(Types.INT())])).add_sink(self.test_sink)
        self.env.execute('test_collection_type_info')
        results = self.test_sink.get_results()
        expected = ['+I[1, [1.1, 1.2, 1.3], [null, hi, flink], 2021-01-09, 12:00:00, 2021-01-09 12:00:00.011, [1, 2, 3]]']
        self.assert_equals_sorted(expected, results)

    def test_process_function(self):
        if False:
            return 10
        self.env.set_parallelism(1)
        self.env.get_config().set_auto_watermark_interval(2000)
        self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
        data_stream = self.env.from_collection([(1, '1603708211000'), (2, '1603708224000'), (3, '1603708226000'), (4, '1603708289000')], type_info=Types.ROW([Types.INT(), Types.STRING()]))

        class MyProcessFunction(ProcessFunction):

            def process_element(self, value, ctx):
                if False:
                    while True:
                        i = 10
                current_timestamp = ctx.timestamp()
                yield 'current timestamp: {}, current_value: {}'.format(str(current_timestamp), str(value))
        watermark_strategy = WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(SecondColumnTimestampAssigner())
        data_stream.assign_timestamps_and_watermarks(watermark_strategy).process(MyProcessFunction(), output_type=Types.STRING()).add_sink(self.test_sink)
        self.env.execute('test process function')
        results = self.test_sink.get_results()
        expected = ["current timestamp: 1603708211000, current_value: Row(f0=1, f1='1603708211000')", "current timestamp: 1603708224000, current_value: Row(f0=2, f1='1603708224000')", "current timestamp: 1603708226000, current_value: Row(f0=3, f1='1603708226000')", "current timestamp: 1603708289000, current_value: Row(f0=4, f1='1603708289000')"]
        self.assert_equals_sorted(expected, results)

class ProcessDataStreamStreamingTests(DataStreamStreamingTests, ProcessDataStreamTests, PyFlinkStreamingTestCase):

    def test_keyed_sum(self):
        if False:
            return 10
        self.env.set_parallelism(1)
        ds = self.env.from_collection([(1, 1), (1, 2), (1, 3), (2, 5), (2, 1)], type_info=Types.ROW_NAMED(['v1', 'v2'], [Types.INT(), Types.INT()]))
        ds.key_by(lambda x: x[0]).sum('v2').key_by(lambda x: x[0]).sum(1).map(lambda x: (x[1], x[0]), output_type=Types.TUPLE([Types.INT(), Types.INT()])).key_by(lambda x: x[1]).sum().add_sink(self.test_sink)
        self.env.execute('key_by_sum_test_stream')
        results = self.test_sink.get_results(False)
        expected = ['(1,1)', '(5,1)', '(15,1)', '(5,2)', '(16,2)']
        self.assert_equals_sorted(expected, results)

    def test_keyed_min_by_and_max(self):
        if False:
            while True:
                i = 10
        self.env.set_parallelism(1)
        ds = self.env.from_collection([('a', 3, 0), ('a', 1, 1), ('b', 5, 0), ('b', 3, 1)], type_info=Types.ROW_NAMED(['v1', 'v2', 'v3'], [Types.STRING(), Types.INT(), Types.INT()]))
        ds.key_by(lambda x: x[0]).min_by('v2').map(lambda x: (x[0], x[1], x[2]), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.INT()])).key_by(lambda x: x[2]).max(1).key_by(lambda x: x[2]).min().map(lambda x: x[0], output_type=Types.STRING()).key_by(lambda x: x).max_by().add_sink(self.test_sink)
        self.env.execute('key_by_min_by_max_by_test_stream')
        results = self.test_sink.get_results(False)
        expected = ['a', 'a', 'a', 'a']
        self.assert_equals_sorted(expected, results)

class ProcessDataStreamBatchTests(DataStreamBatchTests, ProcessDataStreamTests, PyFlinkBatchTestCase):

    def test_keyed_sum(self):
        if False:
            for i in range(10):
                print('nop')
        self.env.set_parallelism(1)
        ds = self.env.from_collection([(1, 1), (1, 2), (1, 3), (5, 1), (5, 5)], type_info=Types.ROW_NAMED(['v1', 'v2'], [Types.INT(), Types.INT()]))

        def flat_map_func1(data):
            if False:
                return 10
            for i in data:
                yield (12, i)

        def flat_map_func2(data):
            if False:
                for i in range(10):
                    print('nop')
            for i in data:
                yield i
        ds.key_by(lambda x: x[0]).sum('v2').key_by(lambda x: x[1]).sum().flat_map(flat_map_func1, output_type=Types.TUPLE([Types.INT(), Types.INT()])).key_by(lambda x: x[0]).sum(1).flat_map(flat_map_func2, output_type=Types.INT()).key_by(lambda x: x).sum().add_sink(self.test_sink)
        self.env.execute('key_by_sum_test_batch')
        results = self.test_sink.get_results(False)
        expected = ['24']
        self.assertEqual(expected, results)

    def test_keyed_min_by_and_max(self):
        if False:
            i = 10
            return i + 15
        self.env.set_parallelism(1)
        ds = self.env.from_collection([(1, '9', 0), (1, '5', 1), (1, '6', 2), (5, '5', 0), (5, '3', 1)], type_info=Types.ROW_NAMED(['v1', 'v2', 'v3'], [Types.INT(), Types.STRING(), Types.INT()]))

        def flat_map_func1(data):
            if False:
                return 10
            for i in data:
                yield (int(i), 1)

        def flat_map_func2(data):
            if False:
                return 10
            for i in data:
                yield i
        ds.key_by(lambda x: x[0]).min_by('v2').map(lambda x: (x[0], x[1], x[2]), output_type=Types.TUPLE([Types.INT(), Types.STRING(), Types.INT()])).key_by(lambda x: x[2]).max(0).flat_map(flat_map_func1, output_type=Types.TUPLE([Types.INT(), Types.INT()])).key_by(lambda x: [1]).min_by().flat_map(flat_map_func2, output_type=Types.INT()).key_by(lambda x: x).max_by().add_sink(self.test_sink)
        self.env.execute('key_by_min_by_max_by_test_batch')
        results = self.test_sink.get_results(False)
        expected = ['1']
        self.assert_equals_sorted(expected, results)

class EmbeddedDataStreamStreamTests(DataStreamStreamingTests, PyFlinkStreamingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(EmbeddedDataStreamStreamTests, self).setUp()
        config = get_j_env_configuration(self.env._j_stream_execution_environment)
        config.setString('python.execution-mode', 'thread')

    def test_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        ds = self.env.from_collection([('ab', 'a', decimal.Decimal(1)), ('bdc', 'a', decimal.Decimal(2)), ('cfgs', 'a', decimal.Decimal(3)), ('deeefg', 'a', decimal.Decimal(4))], type_info=Types.TUPLE([Types.STRING(), Types.STRING(), Types.BIG_DEC()]))

        class MyMapFunction(MapFunction):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.counter = None
                self.counter_value = 0
                self.meter = None
                self.meter_value = 0
                self.value_to_expose = 0
                self.distribution = None

            def open(self, runtime_context: RuntimeContext):
                if False:
                    for i in range(10):
                        print('nop')
                self.counter = runtime_context.get_metrics_group().counter('my_counter')
                self.meter = runtime_context.get_metrics_group().meter('my_meter', 1)
                runtime_context.get_metrics_group().gauge('my_gauge', lambda : self.value_to_expose)
                self.distribution = runtime_context.get_metrics_group().distribution('my_distribution')

            def map(self, value):
                if False:
                    while True:
                        i = 10
                self.counter.inc()
                self.counter_value += 1
                assert self.counter.get_count() == self.counter_value
                self.meter.mark_event(1)
                self.meter_value += 1
                assert self.meter.get_count() == self.meter_value
                self.value_to_expose += 1
                self.distribution.update(int(value[2]))
                return Row(value[0], len(value[0]), value[2])
        ds.key_by(lambda value: value[1]).map(MyMapFunction(), output_type=Types.ROW([Types.STRING(), Types.INT(), Types.BIG_DEC()])).add_sink(self.test_sink)
        self.env.execute('test_basic_operations')
        results = self.test_sink.get_results()
        expected = ['+I[ab, 2, 1]', '+I[bdc, 3, 2]', '+I[cfgs, 4, 3]', '+I[deeefg, 6, 4]']
        self.assert_equals_sorted(expected, results)

class EmbeddedDataStreamBatchTests(DataStreamBatchTests, PyFlinkBatchTestCase):

    def setUp(self):
        if False:
            return 10
        super(EmbeddedDataStreamBatchTests, self).setUp()
        config = get_j_env_configuration(self.env._j_stream_execution_environment)
        config.setString('python.execution-mode', 'thread')

class CommonDataStreamTests(PyFlinkTestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(CommonDataStreamTests, self).setUp()
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_parallelism(2)
        self.env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
        config = get_j_env_configuration(self.env._j_stream_execution_environment)
        config.setString('pekko.ask.timeout', '20 s')
        self.test_sink = DataStreamTestSinkFunction()

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        self.test_sink.clear()

    def assert_equals_sorted(self, expected, actual):
        if False:
            for i in range(10):
                print('nop')
        expected.sort(key=lambda x: str(x))
        actual.sort(key=lambda x: str(x))
        self.assertEqual(expected, actual)

    def test_data_stream_name(self):
        if False:
            i = 10
            return i + 15
        ds = self.env.from_collection([(1, 'Hi', 'Hello'), (2, 'Hello', 'Hi')])
        test_name = 'test_name'
        ds.name(test_name)
        self.assertEqual(test_name, ds.get_name())

    def test_set_parallelism(self):
        if False:
            while True:
                i = 10
        parallelism = 3
        ds = self.env.from_collection([(1, 'Hi', 'Hello'), (2, 'Hello', 'Hi')]).map(lambda x: x)
        ds.set_parallelism(parallelism).add_sink(self.test_sink)
        plan = eval(str(self.env.get_execution_plan()))
        self.assertEqual(parallelism, plan['nodes'][1]['parallelism'])

    def test_set_max_parallelism(self):
        if False:
            i = 10
            return i + 15
        max_parallelism = 4
        self.env.set_parallelism(8)
        ds = self.env.from_collection([(1, 'Hi', 'Hello'), (2, 'Hello', 'Hi')]).map(lambda x: x)
        ds.set_parallelism(max_parallelism).add_sink(self.test_sink)
        plan = eval(str(self.env.get_execution_plan()))
        self.assertEqual(max_parallelism, plan['nodes'][1]['parallelism'])

    def test_force_non_parallel(self):
        if False:
            i = 10
            return i + 15
        self.env.set_parallelism(8)
        ds = self.env.from_collection([(1, 'Hi', 'Hello'), (2, 'Hello', 'Hi')])
        ds.force_non_parallel().add_sink(self.test_sink)
        plan = eval(str(self.env.get_execution_plan()))
        self.assertEqual(1, plan['nodes'][0]['parallelism'])

    def test_union(self):
        if False:
            i = 10
            return i + 15
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_2 = self.env.from_collection([4, 5, 6])
        ds_3 = self.env.from_collection([7, 8, 9])
        unioned_stream = ds_3.union(ds_1, ds_2)
        unioned_stream.map(lambda x: x + 1).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        source_ids = []
        union_node_pre_ids = []
        for node in exec_plan['nodes']:
            if node['pact'] == 'Data Source':
                source_ids.append(node['id'])
            if node['pact'] == 'Operator':
                for pre in node['predecessors']:
                    union_node_pre_ids.append(pre['id'])
        source_ids.sort()
        union_node_pre_ids.sort()
        self.assertEqual(source_ids, union_node_pre_ids)

    def test_keyed_stream_union(self):
        if False:
            i = 10
            return i + 15
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_2 = self.env.from_collection([4, 5, 6])
        unioned_stream = ds_1.key_by(lambda x: x).union(ds_2.key_by(lambda x: x))
        unioned_stream.add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        expected_union_node_pre_ids = []
        union_node_pre_ids = []
        for node in exec_plan['nodes']:
            if node['type'] == '_keyed_stream_values_operator':
                expected_union_node_pre_ids.append(node['id'])
            if node['pact'] == 'Data Sink':
                for pre in node['predecessors']:
                    union_node_pre_ids.append(pre['id'])
        expected_union_node_pre_ids.sort()
        union_node_pre_ids.sort()
        self.assertEqual(expected_union_node_pre_ids, union_node_pre_ids)

    def test_project(self):
        if False:
            print('Hello World!')
        ds = self.env.from_collection([[1, 2, 3, 4], [5, 6, 7, 8]], type_info=Types.TUPLE([Types.INT(), Types.INT(), Types.INT(), Types.INT()]))
        ds.project(1, 3).map(lambda x: (x[0], x[1] + 1)).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        self.assertEqual(exec_plan['nodes'][1]['type'], 'Projection')

    def test_broadcast(self):
        if False:
            for i in range(10):
                print('nop')
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_1.broadcast().map(lambda x: x + 1).set_parallelism(3).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        broadcast_node = exec_plan['nodes'][1]
        pre_ship_strategy = broadcast_node['predecessors'][0]['ship_strategy']
        self.assertEqual(pre_ship_strategy, 'BROADCAST')

    def test_rebalance(self):
        if False:
            i = 10
            return i + 15
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_1.rebalance().map(lambda x: x + 1).set_parallelism(3).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        rebalance_node = exec_plan['nodes'][1]
        pre_ship_strategy = rebalance_node['predecessors'][0]['ship_strategy']
        self.assertEqual(pre_ship_strategy, 'REBALANCE')

    def test_rescale(self):
        if False:
            return 10
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_1.rescale().map(lambda x: x + 1).set_parallelism(3).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        rescale_node = exec_plan['nodes'][1]
        pre_ship_strategy = rescale_node['predecessors'][0]['ship_strategy']
        self.assertEqual(pre_ship_strategy, 'RESCALE')

    def test_shuffle(self):
        if False:
            i = 10
            return i + 15
        ds_1 = self.env.from_collection([1, 2, 3])
        ds_1.shuffle().map(lambda x: x + 1).set_parallelism(3).add_sink(self.test_sink)
        exec_plan = eval(self.env.get_execution_plan())
        shuffle_node = exec_plan['nodes'][1]
        pre_ship_strategy = shuffle_node['predecessors'][0]['ship_strategy']
        self.assertEqual(pre_ship_strategy, 'SHUFFLE')

    def test_keyed_stream_partitioning(self):
        if False:
            i = 10
            return i + 15
        ds = self.env.from_collection([('ab', 1), ('bdc', 2), ('cfgs', 3), ('deeefg', 4)])
        keyed_stream = ds.key_by(lambda x: x[1])
        with self.assertRaises(Exception):
            keyed_stream.shuffle()
        with self.assertRaises(Exception):
            keyed_stream.rebalance()
        with self.assertRaises(Exception):
            keyed_stream.rescale()
        with self.assertRaises(Exception):
            keyed_stream.broadcast()
        with self.assertRaises(Exception):
            keyed_stream.forward()

    def test_slot_sharing_group(self):
        if False:
            for i in range(10):
                print('nop')
        source_operator_name = 'collection source'
        map_operator_name = 'map_operator'
        slot_sharing_group_1 = 'slot_sharing_group_1'
        slot_sharing_group_2 = 'slot_sharing_group_2'
        ds_1 = self.env.from_collection([1, 2, 3]).name(source_operator_name)
        ds_1.slot_sharing_group(SlotSharingGroup.builder(slot_sharing_group_1).build()).map(lambda x: x + 1).set_parallelism(3).name(map_operator_name).slot_sharing_group(slot_sharing_group_2).add_sink(self.test_sink)
        j_generated_stream_graph = self.env._j_stream_execution_environment.getStreamGraph(True)
        j_stream_nodes = list(j_generated_stream_graph.getStreamNodes().toArray())
        for j_stream_node in j_stream_nodes:
            if j_stream_node.getOperatorName() == source_operator_name:
                self.assertEqual(j_stream_node.getSlotSharingGroup(), slot_sharing_group_1)
            elif j_stream_node.getOperatorName() == map_operator_name:
                self.assertEqual(j_stream_node.getSlotSharingGroup(), slot_sharing_group_2)

    def test_chaining_strategy(self):
        if False:
            print('Hello World!')
        chained_operator_name_0 = 'map_operator_0'
        chained_operator_name_1 = 'map_operator_1'
        chained_operator_name_2 = 'map_operator_2'
        ds = self.env.from_collection([1, 2, 3])
        ds.map(lambda x: x).set_parallelism(2).name(chained_operator_name_0).map(lambda x: x).set_parallelism(2).name(chained_operator_name_1).map(lambda x: x).set_parallelism(2).name(chained_operator_name_2).add_sink(self.test_sink)

        def assert_chainable(j_stream_graph, expected_upstream_chainable, expected_downstream_chainable):
            if False:
                for i in range(10):
                    print('nop')
            j_stream_nodes = list(j_stream_graph.getStreamNodes().toArray())
            for j_stream_node in j_stream_nodes:
                if j_stream_node.getOperatorName() == chained_operator_name_1:
                    JStreamingJobGraphGenerator = get_gateway().jvm.org.apache.flink.streaming.api.graph.StreamingJobGraphGenerator
                    j_in_stream_edge = j_stream_node.getInEdges().get(0)
                    upstream_chainable = JStreamingJobGraphGenerator.isChainable(j_in_stream_edge, j_stream_graph)
                    self.assertEqual(expected_upstream_chainable, upstream_chainable)
                    j_out_stream_edge = j_stream_node.getOutEdges().get(0)
                    downstream_chainable = JStreamingJobGraphGenerator.isChainable(j_out_stream_edge, j_stream_graph)
                    self.assertEqual(expected_downstream_chainable, downstream_chainable)
        j_generated_stream_graph = self.env._j_stream_execution_environment.getStreamGraph(True)
        assert_chainable(j_generated_stream_graph, True, True)
        ds = self.env.from_collection([1, 2, 3])
        ds.map(lambda x: x).set_parallelism(2).name(chained_operator_name_0).map(lambda x: x).set_parallelism(2).name(chained_operator_name_1).start_new_chain().map(lambda x: x).set_parallelism(2).name(chained_operator_name_2).add_sink(self.test_sink)
        j_generated_stream_graph = self.env._j_stream_execution_environment.getStreamGraph(True)
        assert_chainable(j_generated_stream_graph, False, True)
        ds = self.env.from_collection([1, 2, 3])
        ds.map(lambda x: x).set_parallelism(2).name(chained_operator_name_0).map(lambda x: x).set_parallelism(2).name(chained_operator_name_1).disable_chaining().map(lambda x: x).set_parallelism(2).name(chained_operator_name_2).add_sink(self.test_sink)
        j_generated_stream_graph = self.env._j_stream_execution_environment.getStreamGraph(True)
        assert_chainable(j_generated_stream_graph, False, False)

    def test_execute_and_collect(self):
        if False:
            i = 10
            return i + 15
        test_data = ['pyflink', 'datastream', 'execute', 'collect']
        ds = self.env.from_collection(test_data)
        expected = test_data[:3]
        actual = []
        for result in ds.execute_and_collect(limit=3):
            actual.append(result)
        self.assertEqual(expected, actual)
        test_data = [('pyflink', 1), ('datastream', 2), ('pyflink', 1), ('collect', 2)]
        expected = [Row(f0='pyflink', f1=('pyflink', 1)), Row(f0='datastream', f1=('datastream', 2)), Row(f0='pyflink', f1=('pyflink', 1)), Row(f0='collect', f1=('collect', 2))]
        ds = self.env.from_collection(collection=test_data, type_info=Types.TUPLE([Types.STRING(), Types.INT()]))
        with ds.key_by(lambda i: i[0], Types.STRING()).execute_and_collect() as results:
            actual = []
            for result in results:
                actual.append(result)
            self.assertEqual(expected, actual)
        test_data = [(1, None, 1, True, 32767, -2147483648, 1.23, 1.98932, bytearray(b'flink'), 'pyflink', datetime.date(2014, 9, 13), datetime.time(hour=12, minute=0, second=0, microsecond=123000), datetime.datetime(2018, 3, 11, 3, 0, 0, 123000), [1, 2, 3], [['pyflink', 'datastream'], ['execute', 'collect']], decimal.Decimal('1000000000000000000.05'), decimal.Decimal('1000000000000000000.05999999999999999899999999999')), (2, None, 2, True, 23878, 652516352, 9.87, 2.98936, bytearray(b'flink'), 'pyflink', datetime.date(2015, 10, 14), datetime.time(hour=11, minute=2, second=2, microsecond=234500), datetime.datetime(2020, 4, 15, 8, 2, 6, 235000), [2, 4, 6], [['pyflink', 'datastream'], ['execute', 'collect']], decimal.Decimal('2000000000000000000.74'), decimal.Decimal('2000000000000000000.06111111111111111111111111111'))]
        expected = test_data
        ds = self.env.from_collection(test_data)
        with ds.execute_and_collect() as results:
            actual = [result for result in results]
            self.assert_equals_sorted(expected, actual)
        test_data = [[1, 2, 3], [4, 5]]
        expected = test_data
        ds = self.env.from_collection(test_data, type_info=Types.PRIMITIVE_ARRAY(Types.INT()))
        with ds.execute_and_collect() as results:
            actual = [r for r in results]
            self.assert_equals_sorted(expected, actual)
        test_data = [(['test', 'test'], [0.0, 0.0]), ([None], [0.0, 0.0])]
        ds = self.env.from_collection(test_data, type_info=Types.TUPLE([Types.OBJECT_ARRAY(Types.STRING()), Types.OBJECT_ARRAY(Types.DOUBLE())]))
        expected = test_data
        with ds.execute_and_collect() as results:
            actual = [result for result in results]
            self.assert_equals_sorted(expected, actual)

    def test_function_with_error(self):
        if False:
            i = 10
            return i + 15
        ds = self.env.from_collection([('a', 0), ('b', 0), ('c', 1), ('d', 1), ('e', 1)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        keyed_stream = ds.key_by(MyKeySelector(), key_type=Types.INT())

        def flat_map_func(x):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('flat_map_func error')
            yield x
        from py4j.protocol import Py4JJavaError
        import pytest
        with pytest.raises(Py4JJavaError, match='flat_map_func error'):
            keyed_stream.flat_map(flat_map_func).print()
            self.env.execute('test_process_function_with_error')

    def test_data_with_custom_class(self):
        if False:
            return 10

        class Data(object):

            def __init__(self, name, num):
                if False:
                    i = 10
                    return i + 15
                self.name = name
                self.num = num
        ds = self.env.from_collection([('a', 0), ('b', 1), ('c', 2)], type_info=Types.ROW([Types.STRING(), Types.INT()]))
        ds.map(lambda a: Data(a[0], a[1])).flat_map(lambda data: [data.name for _ in range(data.num)]).add_sink(self.test_sink)
        self.env.execute('test_data_with_custom_class')
        results = self.test_sink.get_results(True)
        expected = ['c', 'c', 'b']
        self.assert_equals_sorted(expected, results)

class MyKeySelector(KeySelector):

    def get_key(self, value):
        if False:
            print('Hello World!')
        return value[1]

class MyRichCoFlatMapFunction(CoFlatMapFunction):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.map_state = None

    def open(self, runtime_context: RuntimeContext):
        if False:
            i = 10
            return i + 15
        self.map_state = runtime_context.get_map_state(MapStateDescriptor('map', Types.STRING(), Types.BOOLEAN()))

    def flat_map1(self, value):
        if False:
            for i in range(10):
                print('nop')
        yield str(value[0] + 1)
        yield str(value[0] + 1)

    def flat_map2(self, value):
        if False:
            return 10
        if value[0] not in self.map_state:
            self.map_state[value[0]] = True
            yield value[0]

class MyKeyedCoProcessFunction(KeyedCoProcessFunction):

    def __init__(self):
        if False:
            print('Hello World!')
        self.count_state = None
        self.timer_registered = False

    def open(self, runtime_context: RuntimeContext):
        if False:
            while True:
                i = 10
        self.timer_registered = False
        self.count_state = runtime_context.get_state(ValueStateDescriptor('count', Types.INT()))

    def process_element1(self, value, ctx: 'KeyedCoProcessFunction.Context'):
        if False:
            i = 10
            return i + 15
        if not self.timer_registered:
            ctx.timer_service().register_event_time_timer(3)
            self.timer_registered = True
        count = self.count_state.value()
        if count is None:
            count = 1
        else:
            count += 1
        self.count_state.update(count)
        return [Row(value[0], count)]

    def process_element2(self, value, ctx: 'KeyedCoProcessFunction.Context'):
        if False:
            return 10
        count = self.count_state.value()
        if count is None:
            count = 1
        else:
            count += 1
        self.count_state.update(count)
        return [Row(value[0], count)]

    def on_timer(self, timestamp: int, ctx: 'KeyedCoProcessFunction.OnTimerContext'):
        if False:
            print('Hello World!')
        return [Row('on_timer', timestamp)]

class MyReduceFunction(ReduceFunction):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.state = None

    def open(self, runtime_context: RuntimeContext):
        if False:
            i = 10
            return i + 15
        self.state = runtime_context.get_state(ValueStateDescriptor('test_state', Types.INT()))

    def reduce(self, value1, value2):
        if False:
            print('Hello World!')
        state_value = self.state.value()
        if state_value is None:
            state_value = 2
        else:
            state_value += 1
        result_value = Row(value1[0] + value2[0], value1[1])
        if result_value[0] == 'ab':
            assert state_value == 2
        if result_value[0] == 'cde':
            assert state_value == 3
        self.state.update(state_value)
        return result_value

class SecondColumnTimestampAssigner(TimestampAssigner):

    def extract_timestamp(self, value, record_timestamp) -> int:
        if False:
            while True:
                i = 10
        return int(value[1])