"""Worker utilities for representing MapTasks.

Each MapTask represents a sequence of ParallelInstruction(s): read from a
source, write to a sink, parallel do, etc.
"""
import collections
from apache_beam import coders

def build_worker_instruction(*args):
    if False:
        print('Hello World!')
    'Create an object representing a ParallelInstruction protobuf.\n\n  This will be a collections.namedtuple with a custom __str__ method.\n\n  Alas, this wrapper is not known to pylint, which thinks it creates\n  constants.  You may have to put a disable=invalid-name pylint\n  annotation on any use of this, depending on your names.\n\n  Args:\n    *args: first argument is the name of the type to create.  Should\n      start with "Worker".  Second arguments is alist of the\n      attributes of this object.\n  Returns:\n    A new class, a subclass of tuple, that represents the protobuf.\n  '
    tuple_class = collections.namedtuple(*args)
    tuple_class.__str__ = worker_object_to_string
    tuple_class.__repr__ = worker_object_to_string
    return tuple_class

def worker_printable_fields(workerproto):
    if False:
        i = 10
        return i + 15
    'Returns the interesting fields of a Worker* object.'
    return ['%s=%s' % (name, value) for (name, value) in workerproto._asdict().items() if (value or value == 0) and name not in ('coder', 'coders', 'output_coders', 'elements', 'combine_fn', 'serialized_fn', 'window_fn', 'append_trailing_newlines', 'strip_trailing_newlines', 'compression_type', 'context', 'start_shuffle_position', 'end_shuffle_position', 'shuffle_reader_config', 'shuffle_writer_config')]

def worker_object_to_string(worker_object):
    if False:
        print('Hello World!')
    'Returns a string compactly representing a Worker* object.'
    return '%s(%s)' % (worker_object.__class__.__name__, ', '.join(worker_printable_fields(worker_object)))
WorkerRead = build_worker_instruction('WorkerRead', ['source', 'output_coders'])
'Worker details needed to read from a source.\n\nAttributes:\n  source: a source object.\n  output_coders: 1-tuple of the coder for the output.\n'
WorkerSideInputSource = build_worker_instruction('WorkerSideInputSource', ['source', 'tag'])
'Worker details needed to read from a side input source.\n\nAttributes:\n  source: a source object.\n  tag: string tag for this side input.\n'
WorkerGroupingShuffleRead = build_worker_instruction('WorkerGroupingShuffleRead', ['start_shuffle_position', 'end_shuffle_position', 'shuffle_reader_config', 'coder', 'output_coders'])
'Worker details needed to read from a grouping shuffle source.\n\nAttributes:\n  start_shuffle_position: An opaque string to be passed to the shuffle\n    source to indicate where to start reading.\n  end_shuffle_position: An opaque string to be passed to the shuffle\n    source to indicate where to stop reading.\n  shuffle_reader_config: An opaque string used to initialize the shuffle\n    reader. Contains things like connection endpoints for the shuffle\n    server appliance and various options.\n  coder: The KV coder used to decode shuffle entries.\n  output_coders: 1-tuple of the coder for the output.\n'
WorkerUngroupedShuffleRead = build_worker_instruction('WorkerUngroupedShuffleRead', ['start_shuffle_position', 'end_shuffle_position', 'shuffle_reader_config', 'coder', 'output_coders'])
'Worker details needed to read from an ungrouped shuffle source.\n\nAttributes:\n  start_shuffle_position: An opaque string to be passed to the shuffle\n    source to indicate where to start reading.\n  end_shuffle_position: An opaque string to be passed to the shuffle\n    source to indicate where to stop reading.\n  shuffle_reader_config: An opaque string used to initialize the shuffle\n    reader. Contains things like connection endpoints for the shuffle\n    server appliance and various options.\n  coder: The value coder used to decode shuffle entries.\n'
WorkerWrite = build_worker_instruction('WorkerWrite', ['sink', 'input', 'output_coders'])
'Worker details needed to write to a sink.\n\nAttributes:\n  sink: a sink object.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  output_coders: 1-tuple, coder to use to estimate bytes written.\n'
WorkerInMemoryWrite = build_worker_instruction('WorkerInMemoryWrite', ['output_buffer', 'write_windowed_values', 'input', 'output_coders'])
'Worker details needed to write to a in-memory sink.\n\nUsed only for unit testing. It makes worker tests less cluttered with code like\n"write to a file and then check file contents".\n\nAttributes:\n  output_buffer: list to which output elements will be appended\n  write_windowed_values: whether to record the entire WindowedValue outputs,\n    or just the raw (unwindowed) value\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  output_coders: 1-tuple, coder to use to estimate bytes written.\n'
WorkerShuffleWrite = build_worker_instruction('WorkerShuffleWrite', ['shuffle_kind', 'shuffle_writer_config', 'input', 'output_coders'])
"Worker details needed to write to a shuffle sink.\n\nAttributes:\n  shuffle_kind: A string describing the shuffle kind. This can control the\n    way the worker interacts with the shuffle sink. The possible values are:\n    'ungrouped', 'group_keys', and 'group_keys_and_sort_values'.\n  shuffle_writer_config: An opaque string used to initialize the shuffle\n    write. Contains things like connection endpoints for the shuffle\n    server appliance and various options.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  output_coders: 1-tuple of the coder for input elements. If the\n    shuffle_kind is grouping, this is expected to be a KV coder.\n"
WorkerDoFn = build_worker_instruction('WorkerDoFn', ['serialized_fn', 'output_tags', 'input', 'side_inputs', 'output_coders'])
"Worker details needed to run a DoFn.\nAttributes:\n  serialized_fn: A serialized DoFn object to be run for each input element.\n  output_tags: The string tags used to identify the outputs of a ParDo\n    operation. The tag is present even if the ParDo has just one output\n    (e.g., ['out'].\n  output_coders: array of coders, one for each output.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  side_inputs: A list of Worker...Read instances describing sources to be\n    used for getting values. The types supported right now are\n    WorkerInMemoryRead and WorkerTextRead.\n"
WorkerReifyTimestampAndWindows = build_worker_instruction('WorkerReifyTimestampAndWindows', ['output_tags', 'input', 'output_coders'])
"Worker details needed to run a WindowInto.\nAttributes:\n  output_tags: The string tags used to identify the outputs of a ParDo\n    operation. The tag is present even if the ParDo has just one output\n    (e.g., ['out'].\n  output_coders: array of coders, one for each output.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n"
WorkerMergeWindows = build_worker_instruction('WorkerMergeWindows', ['window_fn', 'combine_fn', 'phase', 'output_tags', 'input', 'coders', 'context', 'output_coders'])
"Worker details needed to run a MergeWindows (aka. GroupAlsoByWindows).\nAttributes:\n  window_fn: A serialized Windowing object representing the windowing strategy.\n  combine_fn: A serialized CombineFn object to be used after executing the\n    GroupAlsoByWindows operation. May be None if not a combining operation.\n  phase: Possible values are 'all', 'add', 'merge', and 'extract'.\n    A runner optimizer may split the user combiner in 3 separate\n    phases (ADD, MERGE, and EXTRACT), on separate VMs, as it sees\n    fit. The phase attribute dictates which DoFn is actually running in\n    the worker. May be None if not a combining operation.\n  output_tags: The string tags used to identify the outputs of a ParDo\n    operation. The tag is present even if the ParDo has just one output\n    (e.g., ['out'].\n  output_coders: array of coders, one for each output.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  coders: A 2-tuple of coders (key, value) to encode shuffle entries.\n  context: The ExecutionContext object for the current work item.\n"
WorkerCombineFn = build_worker_instruction('WorkerCombineFn', ['serialized_fn', 'phase', 'input', 'output_coders'])
"Worker details needed to run a CombineFn.\nAttributes:\n  serialized_fn: A serialized CombineFn object to be used.\n  phase: Possible values are 'all', 'add', 'merge', and 'extract'.\n    A runner optimizer may split the user combiner in 3 separate\n    phases (ADD, MERGE, and EXTRACT), on separate VMs, as it sees\n    fit. The phase attribute dictates which DoFn is actually running in\n    the worker.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  output_coders: 1-tuple of the coder for the output.\n"
WorkerPartialGroupByKey = build_worker_instruction('WorkerPartialGroupByKey', ['combine_fn', 'input', 'output_coders'])
'Worker details needed to run a partial group-by-key.\nAttributes:\n  combine_fn: A serialized CombineFn object to be used.\n  input: A (producer index, output index) tuple representing the\n    ParallelInstruction operation whose output feeds into this operation.\n    The output index is 0 except for multi-output operations (like ParDo).\n  output_coders: 1-tuple of the coder for the output.\n'
WorkerFlatten = build_worker_instruction('WorkerFlatten', ['inputs', 'output_coders'])
'Worker details needed to run a Flatten.\nAttributes:\n  inputs: A list of tuples, each (producer index, output index), representing\n    the ParallelInstruction operations whose output feeds into this operation.\n    The output index is 0 unless the input is from a multi-output\n    operation (such as ParDo).\n  output_coders: 1-tuple of the coder for the output.\n'

def get_coder_from_spec(coder_spec):
    if False:
        print('Hello World!')
    "Return a coder instance from a coder spec.\n\n  Args:\n    coder_spec: A dict where the value of the '@type' key is a pickled instance\n      of a Coder instance.\n\n  Returns:\n    A coder instance (has encode/decode methods).\n  "
    assert coder_spec is not None
    ignored_wrappers = 'com.google.cloud.dataflow.sdk.util.TimerOrElement$TimerOrElementCoder'
    if coder_spec['@type'] in ignored_wrappers:
        assert len(coder_spec['component_encodings']) == 1
        coder_spec = coder_spec['component_encodings'][0]
        return get_coder_from_spec(coder_spec)
    if coder_spec['@type'] == 'kind:pair':
        assert len(coder_spec['component_encodings']) == 2
        component_coders = [get_coder_from_spec(c) for c in coder_spec['component_encodings']]
        return coders.TupleCoder(component_coders)
    elif coder_spec['@type'] == 'kind:stream':
        assert len(coder_spec['component_encodings']) == 1
        return coders.IterableCoder(get_coder_from_spec(coder_spec['component_encodings'][0]))
    elif coder_spec['@type'] == 'kind:windowed_value':
        assert len(coder_spec['component_encodings']) == 2
        (value_coder, window_coder) = [get_coder_from_spec(c) for c in coder_spec['component_encodings']]
        return coders.coders.WindowedValueCoder(value_coder, window_coder=window_coder)
    elif coder_spec['@type'] == 'kind:interval_window':
        assert 'component_encodings' not in coder_spec or not coder_spec['component_encodings']
        return coders.coders.IntervalWindowCoder()
    elif coder_spec['@type'] == 'kind:global_window':
        assert 'component_encodings' not in coder_spec or not coder_spec['component_encodings']
        return coders.coders.GlobalWindowCoder()
    elif coder_spec['@type'] == 'kind:varint':
        assert 'component_encodings' not in coder_spec or len(coder_spec['component_encodings'] == 0)
        return coders.coders.VarIntCoder()
    elif coder_spec['@type'] == 'kind:length_prefix':
        assert len(coder_spec['component_encodings']) == 1
        return coders.coders.LengthPrefixCoder(get_coder_from_spec(coder_spec['component_encodings'][0]))
    elif coder_spec['@type'] == 'kind:bytes':
        assert 'component_encodings' not in coder_spec or len(coder_spec['component_encodings'] == 0)
        return coders.BytesCoder()
    return coders.coders.deserialize_coder(coder_spec['@type'].encode('ascii'))