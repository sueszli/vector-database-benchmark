from typing import Set
from typing import Tuple
import apache_beam as beam
from apache_beam.options.pipeline_options import TypeOptions
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms import combiners
from apache_beam.transforms import trigger
from apache_beam.transforms import userstate
from apache_beam.transforms import window
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types

@with_input_types(int)
@with_output_types(int)
class CallSequenceEnforcingCombineFn(beam.CombineFn):
    instances = set()

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._setup_called = False
        self._teardown_called = False

    def setup(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        assert not self._setup_called, 'setup should not be called twice'
        assert not self._teardown_called, 'setup should be called before teardown'
        self.instances.add(self)
        self._setup_called = True

    def create_accumulator(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        assert self._setup_called, 'setup should have been called'
        assert not self._teardown_called, 'teardown should not have been called'
        return 0

    def add_input(self, mutable_accumulator, element, *args, **kwargs):
        if False:
            while True:
                i = 10
        assert self._setup_called, 'setup should have been called'
        assert not self._teardown_called, 'teardown should not have been called'
        mutable_accumulator += element
        return mutable_accumulator

    def add_inputs(self, mutable_accumulator, elements, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.add_input(mutable_accumulator, sum(elements))

    def merge_accumulators(self, accumulators, *args, **kwargs):
        if False:
            return 10
        assert self._setup_called, 'setup should have been called'
        assert not self._teardown_called, 'teardown should not have been called'
        return sum(accumulators)

    def extract_output(self, accumulator, *args, **kwargs):
        if False:
            while True:
                i = 10
        assert self._setup_called, 'setup should have been called'
        assert not self._teardown_called, 'teardown should not have been called'
        return accumulator

    def teardown(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert self._setup_called, 'setup should have been called'
        assert not self._teardown_called, 'teardown should not be called twice'
        self._teardown_called = True

@with_input_types(Tuple[None, str])
@with_output_types(Tuple[int, str])
class IndexAssigningDoFn(beam.DoFn):
    state_param = beam.DoFn.StateParam(userstate.CombiningValueStateSpec('index', beam.coders.VarIntCoder(), CallSequenceEnforcingCombineFn()))

    def process(self, element, state=state_param):
        if False:
            for i in range(10):
                print('nop')
        (_, value) = element
        current_index = state.read()
        yield (current_index, value)
        state.add(1)

def run_combine(pipeline, input_elements=5, lift_combiners=True):
    if False:
        print('Hello World!')
    expected_result = input_elements * (input_elements - 1) / 2
    pipeline.get_pipeline_options().view_as(TypeOptions).runtime_type_check = True
    pipeline.get_pipeline_options().view_as(TypeOptions).allow_unsafe_triggers = True
    with pipeline as p:
        pcoll = p | 'Start' >> beam.Create(range(input_elements))
        if not lift_combiners:
            pcoll |= beam.WindowInto(window.GlobalWindows(), trigger=trigger.AfterCount(input_elements), accumulation_mode=trigger.AccumulationMode.DISCARDING)
        pcoll |= 'Do' >> beam.CombineGlobally(combiners.SingleInputTupleCombineFn(CallSequenceEnforcingCombineFn(), CallSequenceEnforcingCombineFn()), None).with_fanout(fanout=1)
        assert_that(pcoll, equal_to([(expected_result, expected_result)]))

def run_pardo(pipeline, input_elements=10):
    if False:
        for i in range(10):
            print('nop')
    with pipeline as p:
        _ = p | 'Start' >> beam.Create(('Hello' for _ in range(input_elements))) | 'KeyWithNone' >> beam.Map(lambda elem: (None, elem)) | 'Do' >> beam.ParDo(IndexAssigningDoFn())