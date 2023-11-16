import os
import random
import numpy as np
from typing import List, Union, Callable, Optional
from dataclasses import dataclass, field
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali import types
import nvidia.dali.tensors as _Tensors
from test_utils import get_dali_extra_path, check_batch
data_root = get_dali_extra_path()
vid_file = os.path.join(data_root, 'db', 'video', 'sintel', 'sintel_trailer-720p.mp4')

@dataclass
class SampleDesc:
    """Context that the argument provider callback receives when prompted for parameter"""
    rng: random.Random
    frame_idx: int
    sample_idx: int
    batch_idx: int
    sample: np.ndarray

@dataclass
class ArgDesc:
    name: Union[str, int]
    expandable_prefix: str
    dest_device: str
    layout: Optional[str] = None

    def __post_init__(self):
        if False:
            return 10
        assert self.is_positional_arg or self.dest_device != 'gpu', 'Named arguments on GPU are not supported'
        assert not self.layout or self.layout.startswith(self.expandable_prefix)

    @property
    def is_positional_arg(self):
        if False:
            while True:
                i = 10
        return isinstance(self.name, int)

class ArgCb:
    """
    Describes a callback to be used as a per-sample/per-frame argument to the operator.
    ----------
    `name` : Union[str, int]
        String with the name of a named argument of the operator or an int if the data
             should be passed as a positional input.
    `cb` : Callable[[SampleDesc], np.ndarray]
        Callback that based on the SampleDesc instance produces a single parameter for
             specific sample/frame.
    `is_per_frame` : bool
        Flag if the cb should be run for every sample (sequence) or for every frame.
             In the latter case, the argument is passed wrapped
             in per-frame call to the operator.
    `dest_device` : str
        Controls whether the produced data should be passed to the operator in cpu or gpu memory.
        If set to "gpu", the copy to gpu is added in the pipeline.
             Applicable only to positional inputs.
    """

    def __init__(self, name: Union[str, int], cb: Callable[[SampleDesc], np.ndarray], is_per_frame: bool, dest_device: str='cpu'):
        if False:
            for i in range(10):
                print('nop')
        self.desc = ArgDesc(name, 'F' if is_per_frame else '', dest_device)
        self.cb = cb

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ArgCb{}'.format((self.cb, self.desc))

@dataclass
class ArgData:
    desc: ArgDesc
    data: List[List[np.ndarray]] = field(repr=False)

class ParamsProviderBase:
    """
    Computes data to be passed as argument inputs in sequence processing tests, the `compute_params`
    params should return a lists of ArgData describing inputs of the operator, while `expand_params`
    should return corresponding unfolded/expanded ArgData to be used in the baseline pipeline.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.input_data = None
        self.fixed_params = None
        self.rng = None
        self.unfolded_input = None

    def setup(self, input_data: ArgData, fixed_params, rng):
        if False:
            return 10
        self.input_data = input_data
        self.fixed_params = fixed_params
        self.rng = rng

    def unfold_output(self, batches):
        if False:
            print('Hello World!')
        num_expand = len(self.input_data.desc.expandable_prefix)
        return unfold_batch(batches, num_expand)

    def unfold_output_layout(self, layout):
        if False:
            while True:
                i = 10
        num_expand = len(self.input_data.desc.expandable_prefix)
        return layout if not layout else layout[num_expand:]

    def unfold_input(self) -> ArgData:
        if False:
            while True:
                i = 10
        input_desc = self.input_data.desc
        num_expand = len(input_desc.expandable_prefix)
        unfolded_input = unfold_batches(self.input_data.data, num_expand)
        if input_desc.layout:
            unfolded_layout = input_desc.layout[num_expand:]
        else:
            unfolded_layout = input_desc.layout
        self.unfolded_input = ArgData(desc=ArgDesc(input_desc.name, '', input_desc.dest_device, unfolded_layout), data=unfolded_input)
        return self.unfolded_input

    def compute_params(self) -> List[ArgData]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def expand_params(self) -> List[ArgData]:
        if False:
            return 10
        raise NotImplementedError

class ParamsProvider(ParamsProviderBase):

    def __init__(self, input_params: List[ArgCb]):
        if False:
            return 10
        super().__init__()
        self.input_params = input_params
        self.arg_input_data = None
        self.expanded_params_data = None

    def compute_params(self) -> List[ArgData]:
        if False:
            i = 10
            return i + 15
        self.arg_input_data = compute_input_params_data(self.input_data, self.rng, self.input_params)
        return self.arg_input_data

    def expand_params(self) -> List[ArgData]:
        if False:
            i = 10
            return i + 15
        self.expanded_params_data = [ArgData(desc=ArgDesc(arg_data.desc.name, '', arg_data.desc.dest_device), data=expand_arg_input(self.input_data, arg_data)) for arg_data in self.arg_input_data]
        return self.expanded_params_data

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        class_name = repr(self.__class__.__name__).strip("'")
        return f'{class_name}({repr(self.input_params)})'

def arg_data_node(arg_data: ArgData):
    if False:
        for i in range(10):
            print('nop')
    node = fn.external_source(dummy_source(arg_data.data), layout=arg_data.desc.layout)
    if arg_data.desc.dest_device == 'gpu':
        node = node.gpu()
    expandable_prefix = arg_data.desc.expandable_prefix
    if expandable_prefix and expandable_prefix[0] == 'F':
        node = fn.per_frame(node)
    return node

def as_batch(tensor):
    if False:
        i = 10
        return i + 15
    if isinstance(tensor, _Tensors.TensorListGPU):
        tensor = tensor.as_cpu()
    return [np.array(sample, dtype=types.to_numpy_type(sample.dtype)) for sample in tensor]

def dummy_source(batches):
    if False:
        while True:
            i = 10

    def inner():
        if False:
            i = 10
            return i + 15
        while True:
            for batch in batches:
                yield batch
    return inner

def unfold_batch(batch, num_expand):
    if False:
        return 10
    assert num_expand >= 0
    if num_expand == 0:
        return batch
    if num_expand > 1:
        batch = [sample.reshape((-1,) + sample.shape[num_expand:]) for sample in batch]
    return [frame for sample in batch for frame in sample]

def unfold_batches(batches, num_expand):
    if False:
        while True:
            i = 10
    return [unfold_batch(batch, num_expand) for batch in batches]

def get_layout_prefix_len(layout, prefix):
    if False:
        while True:
            i = 10
    for (i, c) in enumerate(layout):
        if c not in prefix:
            return i
    return len(layout)

def expand_arg(expandable_layout, arg_has_frames, input_batch, arg_batch):
    if False:
        i = 10
        return i + 15
    num_expand = len(expandable_layout)
    assert 1 <= num_expand <= 2
    assert all((c in 'FC' for c in expandable_layout))
    assert len(input_batch) == len(arg_batch)
    expanded_batch = []
    for (input_sample, arg_sample) in zip(input_batch, arg_batch):
        if not arg_has_frames or len(arg_sample) == 1:
            arg_sample = arg_sample if not arg_has_frames else arg_sample[0]
            num_frames = np.prod(input_sample.shape[:num_expand])
            expanded_batch.extend((arg_sample for _ in range(num_frames)))
        else:
            frame_idx = expandable_layout.find('F')
            assert frame_idx >= 0
            assert len(arg_sample) == input_sample.shape[frame_idx]
            if num_expand == 1:
                expanded_batch.extend((arg_frame for arg_frame in arg_sample))
            else:
                channel_idx = 1 - frame_idx
                assert expandable_layout[channel_idx] == 'C'
                if channel_idx > frame_idx:
                    expanded_batch.extend((frame_arg for frame_arg in arg_sample for _ in range(input_sample.shape[channel_idx])))
                else:
                    expanded_batch.extend((frame_arg for _ in range(input_sample.shape[channel_idx]) for frame_arg in arg_sample))
    return expanded_batch

def expand_arg_input(input_data: ArgData, arg_data: ArgData):
    if False:
        while True:
            i = 10
    '\n    Expands the `arg_data` to match the sequence shape of input_data.\n    '
    assert arg_data.desc.expandable_prefix in ['F', '']
    assert len(input_data.data) == len(arg_data.data)
    arg_has_frames = arg_data.desc.expandable_prefix == 'F'
    return [expand_arg(input_data.desc.expandable_prefix, arg_has_frames, input_batch, arg_batch) for (input_batch, arg_batch) in zip(input_data.data, arg_data.data)]

def _test_seq_input(num_iters, operator_fn, fixed_params, input_params, input_data: ArgData, rng):
    if False:
        while True:
            i = 10

    @pipeline_def
    def pipeline(args_data: List[ArgData]):
        if False:
            i = 10
            return i + 15
        pos_args = [arg_data for arg_data in args_data if arg_data.desc.is_positional_arg]
        pos_nodes = [None] * len(pos_args)
        for arg_data in pos_args:
            assert 0 <= arg_data.desc.name < len(pos_nodes)
            assert pos_nodes[arg_data.desc.name] is None
            pos_nodes[arg_data.desc.name] = arg_data_node(arg_data)
        named_args = [arg_data for arg_data in args_data if not arg_data.desc.is_positional_arg]
        arg_nodes = {arg_data.desc.name: arg_data_node(arg_data) for arg_data in named_args}
        output = operator_fn(*pos_nodes, **fixed_params, **arg_nodes)
        return output
    assert num_iters >= len(input_data.data)
    max_batch_size = max((len(batch) for batch in input_data.data))
    params_provider = input_params if isinstance(input_params, ParamsProviderBase) else ParamsProvider(input_params)
    params_provider.setup(input_data, fixed_params, rng)
    args_data = params_provider.compute_params()
    seq_pipe = pipeline(args_data=[input_data, *args_data], batch_size=max_batch_size, num_threads=4, device_id=0)
    unfolded_input = params_provider.unfold_input()
    expanded_args_data = params_provider.expand_params()
    max_uf_batch_size = max((len(batch) for batch in unfolded_input.data))
    baseline_pipe = pipeline(args_data=[unfolded_input, *expanded_args_data], batch_size=max_uf_batch_size, num_threads=4, device_id=0)
    seq_pipe.build()
    baseline_pipe.build()
    for _ in range(num_iters):
        (seq_batch,) = seq_pipe.run()
        (baseline_batch,) = baseline_pipe.run()
        assert params_provider.unfold_output_layout(seq_batch.layout()) == baseline_batch.layout()
        batch = params_provider.unfold_output(as_batch(seq_batch))
        baseline_batch = as_batch(baseline_batch)
        assert len(batch) == len(baseline_batch)
        check_batch(batch, baseline_batch, len(batch))

def get_input_arg_per_sample(input_data, param_cb, rng):
    if False:
        for i in range(10):
            print('nop')
    return [[param_cb(SampleDesc(rng, None, sample_idx, batch_idx, sample)) for (sample_idx, sample) in enumerate(batch)] for (batch_idx, batch) in enumerate(input_data.data)]

def get_input_arg_per_frame(input_data: ArgData, param_cb, rng, check_broadcasting):
    if False:
        return 10
    frame_idx = input_data.desc.expandable_prefix.find('F')
    assert frame_idx >= 0

    def arg_for_sample(sample_idx, batch_idx, sample):
        if False:
            while True:
                i = 10
        if check_broadcasting and rng.randint(1, 4) == 1:
            return np.array([param_cb(SampleDesc(rng, 0, sample_idx, batch_idx, sample))])
        num_frames = sample.shape[frame_idx]
        return np.array([param_cb(SampleDesc(rng, frame_idx, sample_idx, batch_idx, sample)) for frame_idx in range(num_frames)])
    return [[arg_for_sample(sample_idx, batch_idx, sample) for (sample_idx, sample) in enumerate(batch)] for (batch_idx, batch) in enumerate(input_data.data)]

def compute_input_params_data(input_data: ArgData, rng, input_params: List[ArgCb]):
    if False:
        while True:
            i = 10

    def input_param_data(arg_cb):
        if False:
            print('Hello World!')
        assert arg_cb.desc.expandable_prefix in ['', 'F']
        if arg_cb.desc.expandable_prefix == 'F':
            return get_input_arg_per_frame(input_data, arg_cb.cb, rng, not arg_cb.desc.is_positional_arg)
        return get_input_arg_per_sample(input_data, arg_cb.cb, rng)
    return [ArgData(desc=arg_cb.desc, data=input_param_data(arg_cb)) for arg_cb in input_params]

def sequence_suite_helper(rng, input_cases: List[ArgData], ops_test_cases, num_iters=4):
    if False:
        print('Hello World!')
    '\n    Generates suite of test cases for a sequence processing operator.\n    The operator should meet the SequenceOperator assumptions, i.e.\n    1. process frames (and possibly channels) independently,\n    2. support per-frame tensor arguments.\n    Each test case consists of two pipelines, one fed with the batch of sequences\n    and one fed with the batch of frames, the test compares if the processing of\n    corresponding frames in both pipelines gives the same result. In other words, if\n    given batch = [sequence, ...], the following holds:\n    fn.op([frame for sequence in batch for frame in sequence])\n        == [frame for sequence in fn.op(batch) for frame in sequence]\n    ----------\n    `input_cases`: List[ArgData].\n        Each ArgData instance describes a single parameter (positional or named) that will be\n        passed to the pipeline and serve as a source of truth (regarding the number of expandable\n        dimensions and sequence shape). Based on it, all other inputs defined through ArgCb\n        in `ops_test_cases` will be computed.\n        Note the `.desc.device` argument is ignored in favour of `ops_test_case` devices list.\n    `ops_test_cases` : List[Tuple[\n            Operator,\n            Dict[str, Any],\n            ParamProviderBase|List[ArgCb]]\n        ]]\n        List of operators and their parameters that should be tested.\n        Each element is expected to be a tuple of the form: (\n            fn.operator,\n            {fixed_param_name: fixed_param_value},\n            [ArgCb(tensor_arg_name, single_arg_cb, is_per_frame, dest_device)]\n        )\n        where the first element is ``fn.operator``, the second one is a dictionary of fixed\n        arguments that should be passed to the operator and the third one is a list of ArgCb\n        instances describing tensor input arguments or custom params provider instance\n        (see `ParamsProvider`). The tuple can optionally have fourth element: a list of devices\n        where the main input (from `input_cases`) should be placed.\n    '

    class OpTestCase:

        def __init__(self, operator_fn, fixed_params, input_params, devices=None, input_name=0):
            if False:
                print('Hello World!')
            self.operator_fn = operator_fn
            self.fixed_params = fixed_params
            self.input_params = input_params
            self.devices = ['cpu', 'gpu'] if devices is None else devices
    for test_case_args in ops_test_cases:
        test_case = OpTestCase(*test_case_args)
        for device in test_case.devices:
            for input_case in input_cases:
                input_desc = input_case.desc
                arg_desc = ArgDesc(input_desc.name, input_desc.expandable_prefix, device, input_desc.layout)
                arg_data = ArgData(arg_desc, input_case.data)
                yield (_test_seq_input, num_iters, test_case.operator_fn, test_case.fixed_params, test_case.input_params, arg_data, rng)

def get_video_input_cases(seq_layout, rng, larger_shape=(512, 288), smaller_shape=(384, 216)):
    if False:
        while True:
            i = 10
    max_batch_size = 8
    max_num_frames = 16
    cases = []
    (w, h) = larger_shape
    larger = vid_source(max_batch_size, 1, max_num_frames, w, h, seq_layout)
    (w, h) = smaller_shape
    smaller = vid_source(max_batch_size, 2, max_num_frames, w, h, seq_layout)
    cases.append(smaller)
    samples = [sample for batch in [smaller[0], larger[0], smaller[1]] for sample in batch]
    rng.shuffle(samples)
    case2 = [samples[0:1], samples[1:1 + max_batch_size], samples[1 + max_batch_size:2 * max_batch_size], samples[2 * max_batch_size:3 * max_batch_size]]
    cases.append(case2)
    frames_idx = seq_layout.find('F')
    if frames_idx == 0:
        case3 = [[sample[:rng.randint(1, sample.shape[0])] for sample in batch] for batch in case2]
        cases.append(case3)
    return cases

@pipeline_def
def vid_pipeline(num_frames, width, height, seq_layout):
    if False:
        i = 10
        return i + 15
    (vid, _) = fn.readers.video_resize(filenames=[vid_file], labels=[], name='video reader', sequence_length=num_frames, file_list_include_preceding_frame=True, device='gpu', seed=42, resize_x=width, resize_y=height)
    if seq_layout == 'FCHW':
        vid = fn.transpose(vid, perm=[0, 3, 1, 2])
    elif seq_layout == 'CFHW':
        vid = fn.transpose(vid, perm=[3, 0, 1, 2])
    else:
        assert seq_layout == 'FHWC'
    return vid

def vid_source(batch_size, num_batches, num_frames, width, height, seq_layout):
    if False:
        print('Hello World!')
    pipe = vid_pipeline(num_threads=4, batch_size=batch_size, num_frames=num_frames, width=width, height=height, device_id=0, seq_layout=seq_layout, prefetch_queue_depth=1)
    pipe.build()
    batches = []
    for _ in range(num_batches):
        (pipe_out,) = pipe.run()
        batches.append(as_batch(pipe_out))
    return batches

def video_suite_helper(ops_test_cases, test_channel_first=True, expand_channels=False, rng=None):
    if False:
        print('Hello World!')
    '\n    Generates suite of video test cases for a sequence processing operator.\n    The function prepares video input to be passed as a main input for `sequence_suite_helper`.\n    For testing operator with different input than the video,\n    consider using `sequence_suite_helper` directly.\n    ----------\n    `ops_test_cases` : (see `sequence_suite_helper`).\n    `test_channel_first` : bool\n        If True, the "FCHW" layout is tested.\n    `expand_channels` : bool\n        If True, for the "FCHW" layout the first two (and not just one) dims are expanded,\n        and "CFHW" layout is tested. Requires `test_channel_first` to be True.\n    '
    rng = rng or random.Random(42)
    expandable_extents = 'FC' if expand_channels else 'F'
    layouts = ['FHWC']
    if not test_channel_first:
        assert not expand_channels
    else:
        layouts.append('FCHW')
        if expand_channels:
            layouts.append('CFHW')

    def input_data_desc(layout, input_data):
        if False:
            print('Hello World!')
        num_expand = get_layout_prefix_len(layout, expandable_extents)
        return ArgData(desc=ArgDesc(0, layout[:num_expand], '', layout), data=input_data)
    input_cases = [input_data_desc(input_layout, input_data) for input_layout in layouts for input_data in get_video_input_cases(input_layout, rng)]
    yield from sequence_suite_helper(rng, input_cases, ops_test_cases)