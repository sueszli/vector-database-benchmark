"""Unit tests for the Beam State and Timer API interfaces."""
import unittest
from typing import Any
from typing import List
import mock
import pytest
import apache_beam as beam
from apache_beam.coders import BytesCoder
from apache_beam.coders import ListCoder
from apache_beam.coders import StrUtf8Coder
from apache_beam.coders import VarIntCoder
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.portability import common_urns
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners import pipeline_context
from apache_beam.runners.common import DoFnSignature
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms import trigger
from apache_beam.transforms import userstate
from apache_beam.transforms import window
from apache_beam.transforms.combiners import ToListCombineFn
from apache_beam.transforms.combiners import TopCombineFn
from apache_beam.transforms.core import DoFn
from apache_beam.transforms.timeutil import TimeDomain
from apache_beam.transforms.userstate import BagStateSpec
from apache_beam.transforms.userstate import CombiningValueStateSpec
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.transforms.userstate import SetStateSpec
from apache_beam.transforms.userstate import TimerSpec
from apache_beam.transforms.userstate import get_dofn_specs
from apache_beam.transforms.userstate import is_stateful_dofn
from apache_beam.transforms.userstate import on_timer
from apache_beam.transforms.userstate import validate_stateful_dofn

class TestStatefulDoFn(DoFn):
    """An example stateful DoFn with state and timers."""
    BUFFER_STATE_1 = BagStateSpec('buffer', BytesCoder())
    BUFFER_STATE_2 = BagStateSpec('buffer2', VarIntCoder())
    EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
    EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)
    EXPIRY_TIMER_3 = TimerSpec('expiry3', TimeDomain.WATERMARK)
    EXPIRY_TIMER_FAMILY = TimerSpec('expiry_family', TimeDomain.WATERMARK)

    def process(self, element, t=DoFn.TimestampParam, buffer_1=DoFn.StateParam(BUFFER_STATE_1), buffer_2=DoFn.StateParam(BUFFER_STATE_2), timer_1=DoFn.TimerParam(EXPIRY_TIMER_1), timer_2=DoFn.TimerParam(EXPIRY_TIMER_2), dynamic_timer=DoFn.TimerParam(EXPIRY_TIMER_FAMILY)):
        if False:
            while True:
                i = 10
        yield element

    @on_timer(EXPIRY_TIMER_1)
    def on_expiry_1(self, window=DoFn.WindowParam, timestamp=DoFn.TimestampParam, key=DoFn.KeyParam, buffer=DoFn.StateParam(BUFFER_STATE_1), timer_1=DoFn.TimerParam(EXPIRY_TIMER_1), timer_2=DoFn.TimerParam(EXPIRY_TIMER_2), timer_3=DoFn.TimerParam(EXPIRY_TIMER_3)):
        if False:
            return 10
        yield 'expired1'

    @on_timer(EXPIRY_TIMER_2)
    def on_expiry_2(self, buffer=DoFn.StateParam(BUFFER_STATE_2), timer_2=DoFn.TimerParam(EXPIRY_TIMER_2), timer_3=DoFn.TimerParam(EXPIRY_TIMER_3)):
        if False:
            while True:
                i = 10
        yield 'expired2'

    @on_timer(EXPIRY_TIMER_3)
    def on_expiry_3(self, buffer_1=DoFn.StateParam(BUFFER_STATE_1), buffer_2=DoFn.StateParam(BUFFER_STATE_2), timer_3=DoFn.TimerParam(EXPIRY_TIMER_3)):
        if False:
            print('Hello World!')
        yield 'expired3'

    @on_timer(EXPIRY_TIMER_FAMILY)
    def on_expiry_family(self, dynamic_timer=DoFn.TimerParam(EXPIRY_TIMER_FAMILY), dynamic_timer_tag=DoFn.DynamicTimerTagParam):
        if False:
            while True:
                i = 10
        yield (dynamic_timer_tag, 'expired_dynamic_timer')

class InterfaceTest(unittest.TestCase):

    def _validate_dofn(self, dofn):
        if False:
            return 10
        return DoFnSignature(dofn)

    @mock.patch('apache_beam.transforms.userstate.validate_stateful_dofn')
    def test_validate_dofn(self, unused_mock):
        if False:
            print('Hello World!')
        dofn = TestStatefulDoFn()
        self._validate_dofn(dofn)
        userstate.validate_stateful_dofn.assert_called_with(dofn)

    def test_spec_construction(self):
        if False:
            for i in range(10):
                print('nop')
        BagStateSpec('statename', VarIntCoder())
        with self.assertRaises(TypeError):
            BagStateSpec(123, VarIntCoder())
        CombiningValueStateSpec('statename', VarIntCoder(), TopCombineFn(10))
        with self.assertRaises(TypeError):
            CombiningValueStateSpec(123, VarIntCoder(), TopCombineFn(10))
        with self.assertRaises(TypeError):
            CombiningValueStateSpec('statename', VarIntCoder(), object())
        SetStateSpec('setstatename', VarIntCoder())
        with self.assertRaises(TypeError):
            SetStateSpec(123, VarIntCoder())
        with self.assertRaises(TypeError):
            SetStateSpec('setstatename', object())
        ReadModifyWriteStateSpec('valuestatename', VarIntCoder())
        with self.assertRaises(TypeError):
            ReadModifyWriteStateSpec(123, VarIntCoder())
        with self.assertRaises(TypeError):
            ReadModifyWriteStateSpec('valuestatename', object())
        with self.assertRaises(ValueError):
            DoFn.TimerParam(BagStateSpec('elements', BytesCoder()))
        TimerSpec('timer', TimeDomain.WATERMARK)
        TimerSpec('timer', TimeDomain.REAL_TIME)
        with self.assertRaises(ValueError):
            TimerSpec('timer', 'bogus_time_domain')
        with self.assertRaises(ValueError):
            DoFn.StateParam(TimerSpec('timer', TimeDomain.WATERMARK))

    def test_state_spec_proto_conversion(self):
        if False:
            i = 10
            return i + 15
        context = pipeline_context.PipelineContext()
        state = BagStateSpec('statename', VarIntCoder())
        state_proto = state.to_runner_api(context)
        self.assertEqual(beam_runner_api_pb2.FunctionSpec(urn=common_urns.user_state.BAG.urn), state_proto.protocol)
        context = pipeline_context.PipelineContext()
        state = CombiningValueStateSpec('statename', VarIntCoder(), TopCombineFn(10))
        state_proto = state.to_runner_api(context)
        self.assertEqual(beam_runner_api_pb2.FunctionSpec(urn=common_urns.user_state.BAG.urn), state_proto.protocol)
        context = pipeline_context.PipelineContext()
        state = SetStateSpec('setstatename', VarIntCoder())
        state_proto = state.to_runner_api(context)
        self.assertEqual(beam_runner_api_pb2.FunctionSpec(urn=common_urns.user_state.BAG.urn), state_proto.protocol)
        context = pipeline_context.PipelineContext()
        state = ReadModifyWriteStateSpec('valuestatename', VarIntCoder())
        state_proto = state.to_runner_api(context)
        self.assertEqual(beam_runner_api_pb2.FunctionSpec(urn=common_urns.user_state.BAG.urn), state_proto.protocol)

    def test_param_construction(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            DoFn.StateParam(TimerSpec('timer', TimeDomain.WATERMARK))
        with self.assertRaises(ValueError):
            DoFn.TimerParam(BagStateSpec('elements', BytesCoder()))

    def test_stateful_dofn_detection(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(is_stateful_dofn(DoFn()))
        self.assertTrue(is_stateful_dofn(TestStatefulDoFn()))

    def test_good_signatures(self):
        if False:
            return 10

        class BasicStatefulDoFn(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER = TimerSpec('expiry1', TimeDomain.WATERMARK)
            EXPIRY_TIMER_FAMILY = TimerSpec('expiry_family_1', TimeDomain.WATERMARK)

            def process(self, element, buffer=DoFn.StateParam(BUFFER_STATE), timer1=DoFn.TimerParam(EXPIRY_TIMER), dynamic_timer=DoFn.TimerParam(EXPIRY_TIMER_FAMILY)):
                if False:
                    i = 10
                    return i + 15
                yield element

            @on_timer(EXPIRY_TIMER)
            def expiry_callback(self, element, timer=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    print('Hello World!')
                yield element

            @on_timer(EXPIRY_TIMER_FAMILY)
            def expiry_family_callback(self, element, dynamic_timer=DoFn.TimerParam(EXPIRY_TIMER_FAMILY)):
                if False:
                    while True:
                        i = 10
                yield element
        stateful_dofn = BasicStatefulDoFn()
        signature = self._validate_dofn(stateful_dofn)
        expected_specs = (set([BasicStatefulDoFn.BUFFER_STATE]), set([BasicStatefulDoFn.EXPIRY_TIMER, BasicStatefulDoFn.EXPIRY_TIMER_FAMILY]))
        self.assertEqual(expected_specs, get_dofn_specs(stateful_dofn))
        self.assertEqual(stateful_dofn.expiry_callback, signature.timer_methods[BasicStatefulDoFn.EXPIRY_TIMER].method_value)
        self.assertEqual(stateful_dofn.expiry_family_callback, signature.timer_methods[BasicStatefulDoFn.EXPIRY_TIMER_FAMILY].method_value)
        stateful_dofn = TestStatefulDoFn()
        signature = self._validate_dofn(stateful_dofn)
        expected_specs = (set([TestStatefulDoFn.BUFFER_STATE_1, TestStatefulDoFn.BUFFER_STATE_2]), set([TestStatefulDoFn.EXPIRY_TIMER_1, TestStatefulDoFn.EXPIRY_TIMER_2, TestStatefulDoFn.EXPIRY_TIMER_3, TestStatefulDoFn.EXPIRY_TIMER_FAMILY]))
        self.assertEqual(expected_specs, get_dofn_specs(stateful_dofn))
        self.assertEqual(stateful_dofn.on_expiry_1, signature.timer_methods[TestStatefulDoFn.EXPIRY_TIMER_1].method_value)
        self.assertEqual(stateful_dofn.on_expiry_2, signature.timer_methods[TestStatefulDoFn.EXPIRY_TIMER_2].method_value)
        self.assertEqual(stateful_dofn.on_expiry_3, signature.timer_methods[TestStatefulDoFn.EXPIRY_TIMER_3].method_value)
        self.assertEqual(stateful_dofn.on_expiry_family, signature.timer_methods[TestStatefulDoFn.EXPIRY_TIMER_FAMILY].method_value)

    def test_bad_signatures(self):
        if False:
            for i in range(10):
                print('nop')

        class BadStatefulDoFn1(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())

            def process(self, element, b1=DoFn.StateParam(BUFFER_STATE), b2=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    i = 10
                    return i + 15
                yield element
        with self.assertRaises(ValueError):
            self._validate_dofn(BadStatefulDoFn1())

        class BadStatefulDoFn2(DoFn):
            TIMER = TimerSpec('timer', TimeDomain.WATERMARK)

            def process(self, element, t1=DoFn.TimerParam(TIMER), t2=DoFn.TimerParam(TIMER)):
                if False:
                    while True:
                        i = 10
                yield element
        with self.assertRaises(ValueError):
            self._validate_dofn(BadStatefulDoFn2())

        class BadStatefulDoFn3(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
            EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)

            @on_timer(EXPIRY_TIMER_1)
            def expiry_callback(self, element, b1=DoFn.StateParam(BUFFER_STATE), b2=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    i = 10
                    return i + 15
                yield element
        with self.assertRaises(ValueError):
            self._validate_dofn(BadStatefulDoFn3())

        class BadStatefulDoFn4(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
            EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)

            @on_timer(EXPIRY_TIMER_1)
            def expiry_callback(self, element, t1=DoFn.TimerParam(EXPIRY_TIMER_2), t2=DoFn.TimerParam(EXPIRY_TIMER_2)):
                if False:
                    i = 10
                    return i + 15
                yield element
        with self.assertRaises(ValueError):
            self._validate_dofn(BadStatefulDoFn4())

        class BadStatefulDoFn5(DoFn):
            EXPIRY_TIMER_FAMILY = TimerSpec('dynamic_timer', TimeDomain.WATERMARK)

            def process(self, element, dynamic_timer_1=DoFn.TimerParam(EXPIRY_TIMER_FAMILY), dynamic_timer_2=DoFn.TimerParam(EXPIRY_TIMER_FAMILY)):
                if False:
                    i = 10
                    return i + 15
                yield element
        with self.assertRaises(ValueError):
            self._validate_dofn(BadStatefulDoFn5())

    def test_validation_typos(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Multiple on_timer callbacks registered for TimerSpec\\(.*expiry1\\).'):

            class StatefulDoFnWithTimerWithTypo1(DoFn):
                BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
                EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
                EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)

                def process(self, element):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                @on_timer(EXPIRY_TIMER_1)
                def on_expiry_1(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                    if False:
                        for i in range(10):
                            print('nop')
                    yield 'expired1'

                @on_timer(EXPIRY_TIMER_1)
                def on_expiry_2(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                    if False:
                        print('Hello World!')
                    yield 'expired2'

        class StatefulDoFnWithTimerWithTypo2(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
            EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)

            def process(self, element, timer1=DoFn.TimerParam(EXPIRY_TIMER_1), timer2=DoFn.TimerParam(EXPIRY_TIMER_2)):
                if False:
                    print('Hello World!')
                pass

            @on_timer(EXPIRY_TIMER_1)
            def on_expiry_1(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    while True:
                        i = 10
                yield 'expired1'

            @on_timer(EXPIRY_TIMER_2)
            def on_expiry_1(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    print('Hello World!')
                yield 'expired2'

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return 'StatefulDoFnWithTimerWithTypo2'
        dofn = StatefulDoFnWithTimerWithTypo2()
        with self.assertRaisesRegex(ValueError, 'The on_timer callback for TimerSpec\\(.*expiry1\\) is not the specified .on_expiry_1 method for DoFn StatefulDoFnWithTimerWithTypo2 \\(perhaps it was overwritten\\?\\).'):
            validate_stateful_dofn(dofn)

        class StatefulDoFnWithTimerWithTypo3(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER_1 = TimerSpec('expiry1', TimeDomain.WATERMARK)
            EXPIRY_TIMER_2 = TimerSpec('expiry2', TimeDomain.WATERMARK)

            def process(self, element, timer1=DoFn.TimerParam(EXPIRY_TIMER_1), timer2=DoFn.TimerParam(EXPIRY_TIMER_2)):
                if False:
                    i = 10
                    return i + 15
                pass

            @on_timer(EXPIRY_TIMER_1)
            def on_expiry_1(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    while True:
                        i = 10
                yield 'expired1'

            def on_expiry_2(self, buffer_state=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    return 10
                yield 'expired2'

            def __repr__(self):
                if False:
                    return 10
                return 'StatefulDoFnWithTimerWithTypo3'
        dofn = StatefulDoFnWithTimerWithTypo3()
        with self.assertRaisesRegex(ValueError, 'DoFn StatefulDoFnWithTimerWithTypo3 has a TimerSpec without an associated on_timer callback: TimerSpec\\(.*expiry2\\).'):
            validate_stateful_dofn(dofn)

class StatefulDoFnOnDirectRunnerTest(unittest.TestCase):
    all_records = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        StatefulDoFnOnDirectRunnerTest.all_records = []

    def record_dofn(self):
        if False:
            print('Hello World!')

        class RecordDoFn(DoFn):

            def process(self, element):
                if False:
                    print('Hello World!')
                StatefulDoFnOnDirectRunnerTest.all_records.append(element)
        return RecordDoFn()

    def test_simple_stateful_dofn(self):
        if False:
            print('Hello World!')

        class SimpleTestStatefulDoFn(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            EXPIRY_TIMER = TimerSpec('expiry', TimeDomain.WATERMARK)

            def process(self, element, buffer=DoFn.StateParam(BUFFER_STATE), timer1=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    for i in range(10):
                        print('nop')
                (unused_key, value) = element
                buffer.add(b'A' + str(value).encode('latin1'))
                timer1.set(20)

            @on_timer(EXPIRY_TIMER)
            def expiry_callback(self, buffer=DoFn.StateParam(BUFFER_STATE), timer=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    i = 10
                    return i + 15
                yield b''.join(sorted(buffer.read()))
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1, 2]).add_elements([3]).advance_watermark_to(25).add_elements([4])
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(SimpleTestStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([b'A1A2A3', b'A1A2A3A4'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_clearing_bag_state(self):
        if False:
            while True:
                i = 10

        class BagStateClearingStatefulDoFn(beam.DoFn):
            BAG_STATE = BagStateSpec('bag_state', StrUtf8Coder())
            EMIT_TIMER = TimerSpec('emit_timer', TimeDomain.WATERMARK)
            CLEAR_TIMER = TimerSpec('clear_timer', TimeDomain.WATERMARK)

            def process(self, element, bag_state=beam.DoFn.StateParam(BAG_STATE), emit_timer=beam.DoFn.TimerParam(EMIT_TIMER), clear_timer=beam.DoFn.TimerParam(CLEAR_TIMER)):
                if False:
                    print('Hello World!')
                value = element[1]
                bag_state.add(value)
                clear_timer.set(100)
                emit_timer.set(1000)

            @on_timer(EMIT_TIMER)
            def emit_values(self, bag_state=beam.DoFn.StateParam(BAG_STATE)):
                if False:
                    i = 10
                    return i + 15
                for value in bag_state.read():
                    yield value
                yield 'extra'

            @on_timer(CLEAR_TIMER)
            def clear_values(self, bag_state=beam.DoFn.StateParam(BAG_STATE)):
                if False:
                    while True:
                        i = 10
                bag_state.clear()
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(0).add_elements([('key', 'value')]).advance_watermark_to(100)
            _ = p | test_stream | beam.ParDo(BagStateClearingStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['extra'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_two_timers_one_function(self):
        if False:
            while True:
                i = 10

        class BagStateClearingStatefulDoFn(beam.DoFn):
            BAG_STATE = BagStateSpec('bag_state', StrUtf8Coder())
            EMIT_TIMER = TimerSpec('emit_timer', TimeDomain.WATERMARK)
            EMIT_TWICE_TIMER = TimerSpec('clear_timer', TimeDomain.WATERMARK)

            def process(self, element, bag_state=beam.DoFn.StateParam(BAG_STATE), emit_timer=beam.DoFn.TimerParam(EMIT_TIMER), emit_twice_timer=beam.DoFn.TimerParam(EMIT_TWICE_TIMER)):
                if False:
                    i = 10
                    return i + 15
                value = element[1]
                bag_state.add(value)
                emit_twice_timer.set(100)
                emit_timer.set(1000)

            @on_timer(EMIT_TWICE_TIMER)
            @on_timer(EMIT_TIMER)
            def emit_values(self, bag_state=beam.DoFn.StateParam(BAG_STATE)):
                if False:
                    print('Hello World!')
                for value in bag_state.read():
                    yield value
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(0).add_elements([('key', 'value')]).advance_watermark_to(100)
            _ = p | test_stream | beam.ParDo(BagStateClearingStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['value', 'value'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_simple_read_modify_write_stateful_dofn(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleTestReadModifyWriteStatefulDoFn(DoFn):
            VALUE_STATE = ReadModifyWriteStateSpec('value', StrUtf8Coder())

            def process(self, element, last_element=DoFn.StateParam(VALUE_STATE)):
                if False:
                    print('Hello World!')
                last_element.write('%s:%s' % element)
                yield last_element.read()
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(0).add_elements([('a', 1)]).advance_watermark_to(10).add_elements([('a', 3)]).advance_watermark_to(20).add_elements([('a', 5)])
            p | test_stream | beam.ParDo(SimpleTestReadModifyWriteStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['a:1', 'a:3', 'a:5'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_clearing_read_modify_write_state(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleClearingReadModifyWriteStatefulDoFn(DoFn):
            VALUE_STATE = ReadModifyWriteStateSpec('value', StrUtf8Coder())

            def process(self, element, last_element=DoFn.StateParam(VALUE_STATE)):
                if False:
                    print('Hello World!')
                value = last_element.read()
                if value is not None:
                    yield value
                last_element.clear()
                last_element.write('%s:%s' % (last_element.read(), element[1]))
                if element[1] == 5:
                    yield last_element.read()
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(0).add_elements([('a', 1)]).advance_watermark_to(10).add_elements([('a', 3)]).advance_watermark_to(20).add_elements([('a', 5)])
            p | test_stream | beam.ParDo(SimpleClearingReadModifyWriteStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['None:1', 'None:3', 'None:5'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_simple_set_stateful_dofn(self):
        if False:
            i = 10
            return i + 15

        class SimpleTestSetStatefulDoFn(DoFn):
            BUFFER_STATE = SetStateSpec('buffer', VarIntCoder())
            EXPIRY_TIMER = TimerSpec('expiry', TimeDomain.WATERMARK)

            def process(self, element, buffer=DoFn.StateParam(BUFFER_STATE), timer1=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    for i in range(10):
                        print('nop')
                (unused_key, value) = element
                buffer.add(value)
                timer1.set(20)

            @on_timer(EXPIRY_TIMER)
            def expiry_callback(self, buffer=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    while True:
                        i = 10
                yield sorted(buffer.read())
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1, 2, 3]).add_elements([2]).advance_watermark_to(24)
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(SimpleTestSetStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([[1, 2, 3]], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_clearing_set_state(self):
        if False:
            print('Hello World!')

        class SetStateClearingStatefulDoFn(beam.DoFn):
            SET_STATE = SetStateSpec('buffer', StrUtf8Coder())
            EMIT_TIMER = TimerSpec('emit_timer', TimeDomain.WATERMARK)
            CLEAR_TIMER = TimerSpec('clear_timer', TimeDomain.WATERMARK)

            def process(self, element, set_state=beam.DoFn.StateParam(SET_STATE), emit_timer=beam.DoFn.TimerParam(EMIT_TIMER), clear_timer=beam.DoFn.TimerParam(CLEAR_TIMER)):
                if False:
                    while True:
                        i = 10
                value = element[1]
                set_state.add(value)
                clear_timer.set(100)
                emit_timer.set(1000)

            @on_timer(EMIT_TIMER)
            def emit_values(self, set_state=beam.DoFn.StateParam(SET_STATE)):
                if False:
                    i = 10
                    return i + 15
                for value in set_state.read():
                    yield value

            @on_timer(CLEAR_TIMER)
            def clear_values(self, set_state=beam.DoFn.StateParam(SET_STATE)):
                if False:
                    i = 10
                    return i + 15
                set_state.clear()
                set_state.add('different-value')
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(0).add_elements([('key1', 'value1')]).advance_watermark_to(100)
            _ = p | test_stream | beam.ParDo(SetStateClearingStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['different-value'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_stateful_set_state_portably(self):
        if False:
            print('Hello World!')

        class SetStatefulDoFn(beam.DoFn):
            SET_STATE = SetStateSpec('buffer', VarIntCoder())

            def process(self, element, set_state=beam.DoFn.StateParam(SET_STATE)):
                if False:
                    while True:
                        i = 10
                (_, value) = element
                aggregated_value = 0
                set_state.add(value)
                for saved_value in set_state.read():
                    aggregated_value += saved_value
                yield aggregated_value
        with TestPipeline() as p:
            values = p | beam.Create([('key', 1), ('key', 2), ('key', 3), ('key', 4), ('key', 3)], reshuffle=False)
            actual_values = values | beam.ParDo(SetStatefulDoFn())
            assert_that(actual_values, equal_to([1, 3, 6, 10, 10]))

    def test_stateful_set_state_clean_portably(self):
        if False:
            i = 10
            return i + 15

        class SetStateClearingStatefulDoFn(beam.DoFn):
            SET_STATE = SetStateSpec('buffer', VarIntCoder())
            EMIT_TIMER = TimerSpec('emit_timer', TimeDomain.WATERMARK)

            def process(self, element, set_state=beam.DoFn.StateParam(SET_STATE), emit_timer=beam.DoFn.TimerParam(EMIT_TIMER)):
                if False:
                    for i in range(10):
                        print('nop')
                (_, value) = element
                set_state.add(value)
                all_elements = [element for element in set_state.read()]
                if len(all_elements) == 5:
                    set_state.clear()
                    set_state.add(100)
                    emit_timer.set(1)

            @on_timer(EMIT_TIMER)
            def emit_values(self, set_state=beam.DoFn.StateParam(SET_STATE)):
                if False:
                    for i in range(10):
                        print('nop')
                yield sorted(set_state.read())
        with TestPipeline() as p:
            values = p | beam.Create([('key', 1), ('key', 2), ('key', 3), ('key', 4), ('key', 5)])
            actual_values = values | beam.Map(lambda t: window.TimestampedValue(t, 1)) | beam.WindowInto(window.FixedWindows(1)) | beam.ParDo(SetStateClearingStatefulDoFn())
            assert_that(actual_values, equal_to([[100]]))

    def test_stateful_dofn_nonkeyed_input(self):
        if False:
            for i in range(10):
                print('nop')
        p = TestPipeline()
        values = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'Input elements to the transform .* with stateful DoFn must be key-value pairs.'):
            values | beam.ParDo(TestStatefulDoFn())

    def test_generate_sequence_with_realtime_timer(self):
        if False:
            for i in range(10):
                print('nop')
        from apache_beam.transforms.combiners import CountCombineFn

        class GenerateRecords(beam.DoFn):
            EMIT_TIMER = TimerSpec('emit_timer', TimeDomain.REAL_TIME)
            COUNT_STATE = CombiningValueStateSpec('count_state', VarIntCoder(), CountCombineFn())

            def __init__(self, frequency, total_records):
                if False:
                    print('Hello World!')
                self.total_records = total_records
                self.frequency = frequency

            def process(self, element, emit_timer=beam.DoFn.TimerParam(EMIT_TIMER)):
                if False:
                    for i in range(10):
                        print('nop')
                emit_timer.set(self.frequency)
                yield element[1]

            @on_timer(EMIT_TIMER)
            def emit_values(self, emit_timer=beam.DoFn.TimerParam(EMIT_TIMER), count_state=beam.DoFn.StateParam(COUNT_STATE)):
                if False:
                    return 10
                count = count_state.read() or 0
                if self.total_records == count:
                    return
                count_state.add(1)
                emit_timer.set(count + 1 + self.frequency)
                yield 'value'
        TOTAL_RECORDS = 3
        FREQUENCY = 1
        test_stream = TestStream().advance_watermark_to(0).add_elements([('key', 0)]).advance_processing_time(1).add_elements([('key', 1)]).advance_processing_time(1).add_elements([('key', 2)]).advance_processing_time(1).add_elements([('key', 3)])
        with beam.Pipeline(argv=['--streaming', '--runner=DirectRunner']) as p:
            _ = p | test_stream | beam.ParDo(GenerateRecords(FREQUENCY, TOTAL_RECORDS)) | beam.ParDo(self.record_dofn())
        self.assertEqual([0, 'value', 1, 'value', 2, 'value', 3], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_simple_stateful_dofn_combining(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleTestStatefulDoFn(DoFn):
            BUFFER_STATE = CombiningValueStateSpec('buffer', ListCoder(VarIntCoder()), ToListCombineFn())
            EXPIRY_TIMER = TimerSpec('expiry1', TimeDomain.WATERMARK)

            def process(self, element, buffer=DoFn.StateParam(BUFFER_STATE), timer1=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    i = 10
                    return i + 15
                (unused_key, value) = element
                buffer.add(value)
                timer1.set(20)

            @on_timer(EXPIRY_TIMER)
            def expiry_callback(self, buffer=DoFn.StateParam(BUFFER_STATE), timer=DoFn.TimerParam(EXPIRY_TIMER)):
                if False:
                    return 10
                yield ''.join((str(x) for x in sorted(buffer.read())))
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1, 2]).add_elements([3]).advance_watermark_to(25).add_elements([4])
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(SimpleTestStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual(['123', '1234'], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_timer_output_timestamp(self):
        if False:
            i = 10
            return i + 15

        class TimerEmittingStatefulDoFn(DoFn):
            EMIT_TIMER_1 = TimerSpec('emit1', TimeDomain.WATERMARK)
            EMIT_TIMER_2 = TimerSpec('emit2', TimeDomain.WATERMARK)
            EMIT_TIMER_3 = TimerSpec('emit3', TimeDomain.WATERMARK)

            def process(self, element, timer1=DoFn.TimerParam(EMIT_TIMER_1), timer2=DoFn.TimerParam(EMIT_TIMER_2), timer3=DoFn.TimerParam(EMIT_TIMER_3)):
                if False:
                    print('Hello World!')
                timer1.set(10)
                timer2.set(20)
                timer3.set(30)

            @on_timer(EMIT_TIMER_1)
            def emit_callback_1(self):
                if False:
                    i = 10
                    return i + 15
                yield 'timer1'

            @on_timer(EMIT_TIMER_2)
            def emit_callback_2(self):
                if False:
                    i = 10
                    return i + 15
                yield 'timer2'

            @on_timer(EMIT_TIMER_3)
            def emit_callback_3(self):
                if False:
                    while True:
                        i = 10
                yield 'timer3'

        class TimestampReifyingDoFn(DoFn):

            def process(self, element, ts=DoFn.TimestampParam):
                if False:
                    for i in range(10):
                        print('nop')
                yield (element, int(ts))
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1])
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(TimerEmittingStatefulDoFn()) | beam.ParDo(TimestampReifyingDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('timer1', 10), ('timer2', 20), ('timer3', 30)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_timer_output_timestamp_and_window(self):
        if False:
            print('Hello World!')

        class TimerEmittingStatefulDoFn(DoFn):
            EMIT_TIMER_1 = TimerSpec('emit1', TimeDomain.WATERMARK)

            def process(self, element, timer1=DoFn.TimerParam(EMIT_TIMER_1)):
                if False:
                    i = 10
                    return i + 15
                timer1.set(10)

            @on_timer(EMIT_TIMER_1)
            def emit_callback_1(self, window=DoFn.WindowParam, ts=DoFn.TimestampParam, key=DoFn.KeyParam):
                if False:
                    print('Hello World!')
                yield ('timer1-{key}'.format(key=key), int(ts), int(window.start), int(window.end))
        pipeline_options = PipelineOptions()
        with TestPipeline(options=pipeline_options) as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1])
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | 'window_into' >> beam.WindowInto(window.FixedWindows(5), accumulation_mode=trigger.AccumulationMode.DISCARDING) | beam.ParDo(TimerEmittingStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('timer1-mykey', 10, 10, 15)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_timer_default_tag(self):
        if False:
            while True:
                i = 10

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY = TimerSpec('emit', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_TIMER_FAMILY)):
                if False:
                    print('Hello World!')
                emit.set(10)
                emit.set(20, dynamic_timer_tag='')

            @on_timer(EMIT_TIMER_FAMILY)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    i = 10
                    return i + 15
                yield (tag, ts)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1]).advance_watermark_to_infinity()
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(DynamicTimerDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('', 20)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_dynamic_timer_simple_dofn(self):
        if False:
            for i in range(10):
                print('nop')

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY = TimerSpec('emit', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_TIMER_FAMILY)):
                if False:
                    while True:
                        i = 10
                emit.set(10, dynamic_timer_tag='emit1')
                emit.set(20, dynamic_timer_tag='emit2')
                emit.set(30, dynamic_timer_tag='emit3')

            @on_timer(EMIT_TIMER_FAMILY)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    while True:
                        i = 10
                yield (tag, ts)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([1]).advance_watermark_to_infinity()
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(DynamicTimerDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('emit1', 10), ('emit2', 20), ('emit3', 30)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    @pytest.mark.no_xdist
    @pytest.mark.timeout(10)
    def test_dynamic_timer_clear_then_set_timer(self):
        if False:
            i = 10
            return i + 15

        class EmitTwoEvents(DoFn):
            EMIT_CLEAR_SET_TIMER = TimerSpec('emitclear', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_CLEAR_SET_TIMER)):
                if False:
                    return 10
                yield ('1', 'set')
                emit.set(1)

            @on_timer(EMIT_CLEAR_SET_TIMER)
            def emit_clear(self):
                if False:
                    for i in range(10):
                        print('nop')
                yield ('1', 'clear')

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY = TimerSpec('emit', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_TIMER_FAMILY)):
                if False:
                    for i in range(10):
                        print('nop')
                if element[1] == 'set':
                    emit.set(10, dynamic_timer_tag='emit1')
                    emit.set(20, dynamic_timer_tag='emit2')
                if element[1] == 'clear':
                    emit.set(30, dynamic_timer_tag='emit3')
                    emit.clear(dynamic_timer_tag='emit3')
                    emit.set(40, dynamic_timer_tag='emit3')
                return []

            @on_timer(EMIT_TIMER_FAMILY)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    i = 10
                    return i + 15
                yield (tag, ts)
        with TestPipeline() as p:
            res = p | beam.Create([('1', 'impulse')]) | beam.ParDo(EmitTwoEvents()) | beam.ParDo(DynamicTimerDoFn())
            assert_that(res, equal_to([('emit1', 10), ('emit2', 20), ('emit3', 40)]))

    def test_dynamic_timer_clear_timer(self):
        if False:
            return 10

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY = TimerSpec('emit', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_TIMER_FAMILY)):
                if False:
                    while True:
                        i = 10
                if element[1] == 'set':
                    emit.set(10, dynamic_timer_tag='emit1')
                    emit.set(20, dynamic_timer_tag='emit2')
                    emit.set(30, dynamic_timer_tag='emit3')
                if element[1] == 'clear':
                    emit.clear(dynamic_timer_tag='emit3')

            @on_timer(EMIT_TIMER_FAMILY)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    for i in range(10):
                        print('nop')
                yield (tag, ts)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(5).add_elements(['set']).advance_watermark_to(10).add_elements(['clear']).advance_watermark_to_infinity()
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(DynamicTimerDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('emit1', 10), ('emit2', 20)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_dynamic_timer_multiple(self):
        if False:
            return 10

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY1 = TimerSpec('emit_family_1', TimeDomain.WATERMARK)
            EMIT_TIMER_FAMILY2 = TimerSpec('emit_family_2', TimeDomain.WATERMARK)

            def process(self, element, emit1=DoFn.TimerParam(EMIT_TIMER_FAMILY1), emit2=DoFn.TimerParam(EMIT_TIMER_FAMILY2)):
                if False:
                    return 10
                emit1.set(10, dynamic_timer_tag='emit11')
                emit1.set(20, dynamic_timer_tag='emit12')
                emit1.set(30, dynamic_timer_tag='emit13')
                emit2.set(30, dynamic_timer_tag='emit21')
                emit2.set(20, dynamic_timer_tag='emit22')
                emit2.set(10, dynamic_timer_tag='emit23')

            @on_timer(EMIT_TIMER_FAMILY1)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    for i in range(10):
                        print('nop')
                yield (tag, ts)

            @on_timer(EMIT_TIMER_FAMILY2)
            def emit_callback_2(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    print('Hello World!')
                yield (tag, ts)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(5).add_elements(['1']).advance_watermark_to_infinity()
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(DynamicTimerDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('emit11', 10), ('emit12', 20), ('emit13', 30), ('emit21', 30), ('emit22', 20), ('emit23', 10)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_dynamic_timer_and_simple_timer(self):
        if False:
            print('Hello World!')

        class DynamicTimerDoFn(DoFn):
            EMIT_TIMER_FAMILY = TimerSpec('emit', TimeDomain.WATERMARK)
            GC_TIMER = TimerSpec('gc', TimeDomain.WATERMARK)

            def process(self, element, emit=DoFn.TimerParam(EMIT_TIMER_FAMILY), gc=DoFn.TimerParam(GC_TIMER)):
                if False:
                    print('Hello World!')
                emit.set(10, dynamic_timer_tag='emit1')
                emit.set(20, dynamic_timer_tag='emit2')
                emit.set(30, dynamic_timer_tag='emit3')
                gc.set(40)

            @on_timer(EMIT_TIMER_FAMILY)
            def emit_callback(self, ts=DoFn.TimestampParam, tag=DoFn.DynamicTimerTagParam):
                if False:
                    while True:
                        i = 10
                yield (tag, ts)

            @on_timer(GC_TIMER)
            def gc(self, ts=DoFn.TimestampParam):
                if False:
                    for i in range(10):
                        print('nop')
                yield ('gc', ts)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(5).add_elements(['1']).advance_watermark_to_infinity()
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(DynamicTimerDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('emit1', 10), ('emit2', 20), ('emit3', 30), ('gc', 40)], sorted(StatefulDoFnOnDirectRunnerTest.all_records))

    def test_index_assignment(self):
        if False:
            while True:
                i = 10

        class IndexAssigningStatefulDoFn(DoFn):
            INDEX_STATE = CombiningValueStateSpec('index', sum)

            def process(self, element, state=DoFn.StateParam(INDEX_STATE)):
                if False:
                    return 10
                (unused_key, value) = element
                current_index = state.read()
                yield (value, current_index)
                state.add(1)
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements(['A', 'B']).add_elements(['C']).advance_watermark_to(25).add_elements(['D'])
            p | test_stream | beam.Map(lambda x: ('mykey', x)) | beam.ParDo(IndexAssigningStatefulDoFn()) | beam.ParDo(self.record_dofn())
        self.assertEqual([('A', 0), ('B', 1), ('C', 2), ('D', 3)], StatefulDoFnOnDirectRunnerTest.all_records)

    def test_hash_join(self):
        if False:
            print('Hello World!')

        class HashJoinStatefulDoFn(DoFn):
            BUFFER_STATE = BagStateSpec('buffer', BytesCoder())
            UNMATCHED_TIMER = TimerSpec('unmatched', TimeDomain.WATERMARK)

            def process(self, element, state=DoFn.StateParam(BUFFER_STATE), timer=DoFn.TimerParam(UNMATCHED_TIMER)):
                if False:
                    print('Hello World!')
                (key, value) = element
                existing_values = list(state.read())
                if not existing_values:
                    state.add(value)
                    timer.set(100)
                else:
                    yield (b'Record<%s,%s,%s>' % (key, existing_values[0], value))
                    state.clear()
                    timer.clear()

            @on_timer(UNMATCHED_TIMER)
            def expiry_callback(self, state=DoFn.StateParam(BUFFER_STATE)):
                if False:
                    for i in range(10):
                        print('nop')
                buffered = list(state.read())
                assert len(buffered) == 1, buffered
                state.clear()
                yield (b'Unmatched<%s>' % (buffered[0],))
        with TestPipeline() as p:
            test_stream = TestStream().advance_watermark_to(10).add_elements([(b'A', b'a'), (b'B', b'b')]).add_elements([(b'A', b'aa'), (b'C', b'c')]).advance_watermark_to(25).add_elements([(b'A', b'aaa'), (b'B', b'bb')]).add_elements([(b'D', b'd'), (b'D', b'dd'), (b'D', b'ddd'), (b'D', b'dddd')]).advance_watermark_to(125).add_elements([(b'C', b'cc')])
            p | test_stream | beam.ParDo(HashJoinStatefulDoFn()) | beam.ParDo(self.record_dofn())
        equal_to(StatefulDoFnOnDirectRunnerTest.all_records)([b'Record<A,a,aa>', b'Record<B,b,bb>', b'Record<D,d,dd>', b'Record<D,ddd,dddd>', b'Unmatched<aaa>', b'Unmatched<c>', b'Unmatched<cc>'])
if __name__ == '__main__':
    unittest.main()