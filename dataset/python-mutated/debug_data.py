"""Classes and functions to handle debug-dump data of TensorFlow Debugger."""
import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
METADATA_FILE_PREFIX = '_tfdbg_'
CORE_METADATA_TAG = 'core_metadata_'
GRAPH_FILE_TAG = 'graph_'
DEVICE_TAG = 'device_'
HASH_TAG = 'hash'
FETCHES_INFO_FILE_TAG = 'fetches_info_'
FEED_KEYS_INFO_FILE_TAG = 'feed_keys_info_'

def _glob(glob_pattern):
    if False:
        for i in range(10):
            print('nop')
    if platform.system() == 'Windows':
        return glob.glob(glob_pattern)
    else:
        return gfile.Glob(glob_pattern)

class InconvertibleTensorProto:
    """Represents a TensorProto that cannot be converted to np.ndarray."""

    def __init__(self, tensor_proto, initialized=True):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      tensor_proto: the `TensorProto` object that cannot be represented as a\n        `np.ndarray` object.\n      initialized: (`bool`) whether the Tensor is initialized.\n    '
        self._tensor_proto = tensor_proto
        self._initialized = initialized

    def __str__(self):
        if False:
            print('Hello World!')
        output = '' if self._initialized else 'Uninitialized tensor:\n'
        output += str(self._tensor_proto)
        return output

    @property
    def initialized(self):
        if False:
            print('Hello World!')
        return self._initialized

def load_tensor_from_event_file(event_file_path):
    if False:
        return 10
    'Load a tensor from an event file.\n\n  Assumes that the event file contains a `Event` protobuf and the `Event`\n  protobuf contains a `Tensor` value.\n\n  Args:\n    event_file_path: (`str`) path to the event file.\n\n  Returns:\n    The tensor value loaded from the event file, as a `numpy.ndarray`. For\n    uninitialized Tensors, returns `None`. For Tensors of data types that\n    cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return\n    `None`.\n  '
    event = event_pb2.Event()
    with gfile.Open(event_file_path, 'rb') as f:
        event.ParseFromString(f.read())
        return load_tensor_from_event(event)

def load_tensor_from_event(event):
    if False:
        for i in range(10):
            print('nop')
    'Load a tensor from an Event proto.\n\n  Args:\n    event: The Event proto, assumed to hold a tensor value in its\n        summary.value[0] field.\n\n  Returns:\n    The tensor value loaded from the event file, as a `numpy.ndarray`, if\n    representation of the tensor value by a `numpy.ndarray` is possible.\n    For uninitialized Tensors, returns `None`. For Tensors of data types that\n    cannot be represented as `numpy.ndarray` (e.g., `tf.resource`), return\n    the `TensorProto` protobuf object without converting it to a\n    `numpy.ndarray`.\n  '
    tensor_proto = event.summary.value[0].tensor
    shape = tensor_util.TensorShapeProtoToList(tensor_proto.tensor_shape)
    num_elements = 1
    for shape_dim in shape:
        num_elements *= shape_dim
    if tensor_proto.tensor_content or tensor_proto.string_val or (not num_elements):
        if tensor_proto.dtype == types_pb2.DT_RESOURCE:
            tensor_value = InconvertibleTensorProto(tensor_proto)
        else:
            try:
                tensor_value = tensor_util.MakeNdarray(tensor_proto)
            except KeyError:
                tensor_value = InconvertibleTensorProto(tensor_proto)
    else:
        tensor_value = InconvertibleTensorProto(tensor_proto, False)
    return tensor_value

def _load_graph_def_from_event_file(event_file_path):
    if False:
        return 10
    event = event_pb2.Event()
    with gfile.Open(event_file_path, 'rb') as f:
        event.ParseFromString(f.read())
    return graph_pb2.GraphDef.FromString(event.graph_def)

def _load_log_message_from_event_file(event_file_path):
    if False:
        return 10
    event = event_pb2.Event()
    with gfile.Open(event_file_path, 'rb') as f:
        event.ParseFromString(f.read())
    return event.log_message.message

def _is_graph_file(file_name):
    if False:
        while True:
            i = 10
    return file_name.startswith(METADATA_FILE_PREFIX + GRAPH_FILE_TAG)

def _is_run_fetches_info_file(file_name):
    if False:
        print('Hello World!')
    return file_name == METADATA_FILE_PREFIX + FETCHES_INFO_FILE_TAG

def _is_run_feed_keys_info_file(file_name):
    if False:
        for i in range(10):
            print('nop')
    return file_name == METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG

def _get_tensor_name(node_name, output_slot):
    if False:
        i = 10
        return i + 15
    'Get tensor name given node name and output slot index.\n\n  Args:\n    node_name: Name of the node that outputs the tensor, as a string.\n    output_slot: Output slot index of the tensor, as an integer.\n\n  Returns:\n    Name of the tensor, as a string.\n  '
    return '%s:%d' % (node_name, output_slot)

def _get_tensor_watch_key(node_name, output_slot, debug_op):
    if False:
        i = 10
        return i + 15
    'Get the string representation of a debug watch on a tensor.\n\n  Args:\n    node_name: Name of the node by which the watched tensor is produced, as a\n        string.\n    output_slot: Output slot index of the tensor, as an integer.\n    debug_op: Name of the debug op that is used to watch the tensor, as a\n        string.\n\n  Returns:\n    A string representing the debug watch on the tensor (i.e., the "watch\n        key").\n  '
    return '%s:%s' % (_get_tensor_name(node_name, output_slot), debug_op)

def has_inf_or_nan(datum, tensor):
    if False:
        return 10
    'A predicate for whether a tensor consists of any bad numerical values.\n\n  This predicate is common enough to merit definition in this module.\n  Bad numerical values include `nan`s and `inf`s.\n  The signature of this function follows the requirement of the method\n  `DebugDumpDir.find()`.\n\n  Args:\n    datum: (`DebugTensorDatum`) Datum metadata.\n    tensor: (`numpy.ndarray` or None) Value of the tensor. None represents\n      an uninitialized tensor.\n\n  Returns:\n    (`bool`) True if and only if tensor consists of any nan or inf values.\n  '
    _ = datum
    if isinstance(tensor, InconvertibleTensorProto):
        return False
    elif np.issubdtype(tensor.dtype, np.floating) or np.issubdtype(tensor.dtype, np.complexfloating) or np.issubdtype(tensor.dtype, np.integer):
        return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
    else:
        return False
_CoreMetadata = collections.namedtuple('CoreMetadata', ['global_step', 'session_run_index', 'executor_step_index', 'input_names', 'output_names', 'target_nodes'])

def extract_core_metadata_from_event_proto(event):
    if False:
        i = 10
        return i + 15
    json_metadata = json.loads(event.log_message.message)
    return _CoreMetadata(json_metadata['global_step'], json_metadata['session_run_index'], json_metadata['executor_step_index'], json_metadata['input_names'], json_metadata['output_names'], json_metadata['target_nodes'])

def device_name_to_device_path(device_name):
    if False:
        i = 10
        return i + 15
    'Convert device name to device path.'
    device_name_items = compat.as_text(device_name).split('/')
    device_name_items = [item.replace(':', '_') for item in device_name_items]
    return METADATA_FILE_PREFIX + DEVICE_TAG + ','.join(device_name_items)

def device_path_to_device_name(device_dir):
    if False:
        while True:
            i = 10
    'Parse device name from device path.\n\n  Args:\n    device_dir: (str) a directory name for the device.\n\n  Returns:\n    (str) parsed device name.\n  '
    path_items = os.path.basename(device_dir)[len(METADATA_FILE_PREFIX) + len(DEVICE_TAG):].split(',')
    return '/'.join([path_item.replace('device_', 'device:').replace('_', ':', 1) for path_item in path_items])

class DebugTensorDatum:
    """A single tensor dumped by TensorFlow Debugger (tfdbg).

  Contains metadata about the dumped tensor, including `timestamp`,
  `node_name`, `output_slot`, `debug_op`, and path to the dump file
  (`file_path`).

  This type does not hold the generally space-expensive tensor value (numpy
  array). Instead, it points to the file from which the tensor value can be
  loaded (with the `get_tensor` method) if needed.
  """

    def __init__(self, dump_root, debug_dump_rel_path):
        if False:
            for i in range(10):
                print('nop')
        '`DebugTensorDatum` constructor.\n\n    Args:\n      dump_root: (`str`) Debug dump root directory. This path should not include\n        the path component that represents the device name (see also below).\n      debug_dump_rel_path: (`str`) Path to a debug dump file, relative to the\n        `dump_root`. The first item of this relative path is assumed to be\n        a path representing the name of the device that the Tensor belongs to.\n        See `device_path_to_device_name` for more details on the device path.\n        For example, suppose the debug dump root\n        directory is `/tmp/tfdbg_1` and the dump file is at\n        `/tmp/tfdbg_1/<device_path>/>ns_1/node_a_0_DebugIdentity_123456789`,\n        then the value of the debug_dump_rel_path should be\n        `<device_path>/ns_1/node_a_0_DebugIdentity_1234456789`.\n\n    Raises:\n      ValueError: If the base file name of the dump file does not conform to\n        the dump file naming pattern:\n        `node_name`_`output_slot`_`debug_op`_`timestamp`\n    '
        path_components = os.path.normpath(debug_dump_rel_path).split(os.sep)
        self._device_name = device_path_to_device_name(path_components[0])
        base = path_components[-1]
        if base.count('_') < 3:
            raise ValueError('Dump file path does not conform to the naming pattern: %s' % base)
        self._extended_timestamp = base.split('_')[-1]
        if '-' in self._extended_timestamp:
            self._timestamp = int(self._extended_timestamp[:self._extended_timestamp.find('-')])
        else:
            self._timestamp = int(self._extended_timestamp)
        self._debug_op = base.split('_')[-2]
        self._output_slot = int(base.split('_')[-3])
        node_base_name = '_'.join(base.split('_')[:-3])
        self._node_name = '/'.join(path_components[1:-1] + [node_base_name])
        self._file_path = os.path.join(dump_root, debug_dump_rel_path)
        self._dump_size_bytes = gfile.Stat(self._file_path).length if gfile.Exists(self._file_path) else None

    def __str__(self):
        if False:
            print('Hello World!')
        return '{DebugTensorDatum (%s) %s:%d @ %s @ %d}' % (self.device_name, self.node_name, self.output_slot, self.debug_op, self.timestamp)

    def __repr__(self):
        if False:
            return 10
        return self.__str__()

    def get_tensor(self):
        if False:
            print('Hello World!')
        'Get tensor from the dump (`Event`) file.\n\n    Returns:\n      The tensor loaded from the dump (`Event`) file.\n    '
        return load_tensor_from_event_file(self.file_path)

    @property
    def timestamp(self):
        if False:
            while True:
                i = 10
        'Timestamp of when this tensor value was dumped.\n\n    Returns:\n      (`int`) The timestamp in microseconds.\n    '
        return self._timestamp

    @property
    def extended_timestamp(self):
        if False:
            while True:
                i = 10
        'Extended timestamp, possibly with an index suffix.\n\n    The index suffix, e.g., "-1", is for disambiguating multiple dumps of the\n    same tensor with the same timestamp, which can occur if the dumping events\n    are spaced by shorter than the temporal resolution of the timestamps.\n\n    Returns:\n      (`str`) The extended timestamp.\n    '
        return self._extended_timestamp

    @property
    def debug_op(self):
        if False:
            for i in range(10):
                print('nop')
        'Name of the debug op.\n\n    Returns:\n      (`str`) debug op name (e.g., `DebugIdentity`).\n    '
        return self._debug_op

    @property
    def device_name(self):
        if False:
            i = 10
            return i + 15
        'Name of the device that the tensor belongs to.\n\n    Returns:\n      (`str`) device name.\n    '
        return self._device_name

    @property
    def node_name(self):
        if False:
            print('Hello World!')
        'Name of the node from which the tensor value was dumped.\n\n    Returns:\n      (`str`) name of the node watched by the debug op.\n    '
        return self._node_name

    @property
    def output_slot(self):
        if False:
            print('Hello World!')
        'Output slot index from which the tensor value was dumped.\n\n    Returns:\n      (`int`) output slot index watched by the debug op.\n    '
        return self._output_slot

    @property
    def tensor_name(self):
        if False:
            while True:
                i = 10
        'Name of the tensor watched by the debug op.\n\n    Returns:\n      (`str`) `Tensor` name, in the form of `node_name`:`output_slot`\n    '
        return _get_tensor_name(self.node_name, self.output_slot)

    @property
    def watch_key(self):
        if False:
            return 10
        'Watch key identities a debug watch on a tensor.\n\n    Returns:\n      (`str`) A watch key, in the form of `tensor_name`:`debug_op`.\n    '
        return _get_tensor_watch_key(self.node_name, self.output_slot, self.debug_op)

    @property
    def file_path(self):
        if False:
            print('Hello World!')
        'Path to the file which stores the value of the dumped tensor.'
        return self._file_path

    @property
    def dump_size_bytes(self):
        if False:
            i = 10
            return i + 15
        'Size of the dump file.\n\n    Unit: byte.\n\n    Returns:\n      If the dump file exists, size of the dump file, in bytes.\n      If the dump file does not exist, None.\n    '
        return self._dump_size_bytes

class WatchKeyDoesNotExistInDebugDumpDirError(ValueError):
    pass

class DebugDumpDir:
    """Data set from a debug-dump directory on filesystem.

  An instance of `DebugDumpDir` contains all `DebugTensorDatum` instances
  in a tfdbg dump root directory.
  """

    def __init__(self, dump_root, partition_graphs=None, validate=True):
        if False:
            print('Hello World!')
        '`DebugDumpDir` constructor.\n\n    Args:\n      dump_root: (`str`) path to the dump root directory.\n      partition_graphs: A repeated field of GraphDefs representing the\n          partition graphs executed by the TensorFlow runtime.\n      validate: (`bool`) whether the dump files are to be validated against the\n          partition graphs.\n\n    Raises:\n      IOError: If dump_root does not exist as a directory.\n      ValueError: If more than one core metadata file is found under the dump\n        root directory.\n    '
        if not gfile.IsDirectory(dump_root):
            raise IOError('Dump root directory %s does not exist' % dump_root)
        self._core_metadata = []
        self._dump_root = dump_root
        self._load_core_metadata()
        self._load_fetches_info()
        self._load_feeds_info()
        self._load_all_device_dumps(partition_graphs, validate)
        self._python_graph = None

    def _load_all_device_dumps(self, partition_graphs, validate):
        if False:
            i = 10
            return i + 15
        'Load the dump data for all devices.'
        device_dirs = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + DEVICE_TAG + '*'))
        self._device_names = []
        self._t0s = {}
        self._dump_tensor_data = {}
        self._dump_graph_file_paths = {}
        self._debug_watches = {}
        self._watch_key_to_devices = {}
        self._watch_key_to_datum = {}
        self._watch_key_to_rel_time = {}
        self._watch_key_to_dump_size_bytes = {}
        for device_dir in device_dirs:
            device_name = device_path_to_device_name(device_dir)
            self._device_names.append(device_name)
            self._load_device_dumps(device_name, device_dir)
        self._load_partition_graphs(partition_graphs, validate)
        self._calculate_t0()
        for device_name in self._device_names:
            self._create_tensor_watch_maps(device_name)

    def _load_device_dumps(self, device_name, device_root):
        if False:
            while True:
                i = 10
        'Load `DebugTensorDatum` instances from the dump root of a given device.\n\n    Populates a map {device_name: a list of `DebugTensorDatum`}, where the list\n    is sorted by ascending timestamp.\n\n    This sorting order reflects the order in which the TensorFlow executor\n    processed the nodes of the graph. It is (one of many possible) topological\n    sort of the nodes. This is useful for displaying tensors in the debugger\n    frontend as well as for the use case in which the user wants to find a\n    "culprit tensor", i.e., the first tensor in the graph that exhibits certain\n    problematic properties, i.e., all zero values, or bad numerical values such\n    as nan and inf.\n\n    In addition, creates a map from node name to debug watches. In this Map,\n    the key is the watched node name; the value is a dictionary.\n    Of this dictionary, the key is the watched_output_slot.\n\n    This method attempts to load the debug watches from the tensor dump files\n    first, before loading the full set of debug watches from the partition\n    graphs as done later. This is necessary because sometimes the partition\n    graphs may not be available, e.g., when the run errors out.\n\n    Args:\n      device_name: (`str`) name of the device.\n      device_root: (`str`) dump root directory of the given device.\n\n    Raises:\n      ValueError: If GraphDef for the device is not available.\n    '
        self._dump_tensor_data[device_name] = []
        self._debug_watches[device_name] = collections.defaultdict(lambda : collections.defaultdict(set))
        for (root, _, files) in gfile.Walk(device_root):
            for f in files:
                if _is_graph_file(f):
                    self._dump_graph_file_paths[device_name] = os.path.join(root, f)
                else:
                    datum = self._dump_file_name_to_datum(root, f)
                    self._dump_tensor_data[device_name].append(datum)
                    self._debug_watches[device_name][datum.node_name][datum.output_slot].add(datum.debug_op)
        self._dump_tensor_data[device_name] = sorted(self._dump_tensor_data[device_name], key=lambda x: x.extended_timestamp)
        if self._dump_tensor_data[device_name]:
            self._t0s[device_name] = self._dump_tensor_data[device_name][0].timestamp
        else:
            self._t0s[device_name] = None

    def _calculate_t0(self):
        if False:
            while True:
                i = 10
        'Calculate the first timestamp across all devices.'
        t0s = [t0 for t0 in self._t0s.values() if t0 is not None]
        self._t0 = min(t0s) if t0s else None

    def _load_core_metadata(self):
        if False:
            print('Hello World!')
        core_metadata_files = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + CORE_METADATA_TAG + '*'))
        for core_metadata_file in core_metadata_files:
            with gfile.Open(core_metadata_file, 'rb') as f:
                event = event_pb2.Event()
                event.ParseFromString(f.read())
                self._core_metadata.append(extract_core_metadata_from_event_proto(event))

    def _load_fetches_info(self):
        if False:
            i = 10
            return i + 15
        fetches_info_files = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + FETCHES_INFO_FILE_TAG + '*'))
        self._run_fetches_info = []
        for fetches_info_file in fetches_info_files:
            self._run_fetches_info.append(_load_log_message_from_event_file(fetches_info_file))

    def _load_feeds_info(self):
        if False:
            i = 10
            return i + 15
        feeds_info_files = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG + '*'))
        self._run_feed_keys_info = []
        for feeds_info_file in feeds_info_files:
            self._run_feed_keys_info.append(_load_log_message_from_event_file(feeds_info_file))

    def _dump_file_name_to_datum(self, dir_name, file_name):
        if False:
            i = 10
            return i + 15
        'Obtain a DebugTensorDatum from the directory and file name.\n\n    Args:\n      dir_name: (`str`) Name of the directory in which the dump file resides.\n      file_name: (`str`) Base name of the dump file.\n\n    Returns:\n      (`DebugTensorDatum`) The `DebugTensorDatum` loaded from the dump file.\n    '
        debug_dump_rel_path = os.path.join(os.path.relpath(dir_name, self._dump_root), file_name)
        return DebugTensorDatum(self._dump_root, debug_dump_rel_path)

    def _create_tensor_watch_maps(self, device_name):
        if False:
            i = 10
            return i + 15
        'Create maps from tensor watch keys to datum and to timestamps.\n\n    Create a map from watch key (tensor name + debug op) to `DebugTensorDatum`\n    item. Also make a map from watch key to relative timestamp.\n    "relative" means (absolute timestamp - t0).\n\n    Args:\n      device_name: (str) name of the device.\n    '
        self._watch_key_to_datum[device_name] = {}
        self._watch_key_to_rel_time[device_name] = {}
        self._watch_key_to_dump_size_bytes[device_name] = {}
        for datum in self._dump_tensor_data[device_name]:
            if datum.watch_key not in self._watch_key_to_devices:
                self._watch_key_to_devices[datum.watch_key] = {device_name}
            else:
                self._watch_key_to_devices[datum.watch_key].add(device_name)
            if datum.watch_key not in self._watch_key_to_datum[device_name]:
                self._watch_key_to_datum[device_name][datum.watch_key] = [datum]
                self._watch_key_to_rel_time[device_name][datum.watch_key] = [datum.timestamp - self._t0]
                self._watch_key_to_dump_size_bytes[device_name][datum.watch_key] = [datum.dump_size_bytes]
            else:
                self._watch_key_to_datum[device_name][datum.watch_key].append(datum)
                self._watch_key_to_rel_time[device_name][datum.watch_key].append(datum.timestamp - self._t0)
                self._watch_key_to_dump_size_bytes[device_name][datum.watch_key].append(datum.dump_size_bytes)

    def set_python_graph(self, python_graph):
        if False:
            print('Hello World!')
        'Provide Python `Graph` object to the wrapper.\n\n    Unlike the partition graphs, which are protobuf `GraphDef` objects, `Graph`\n    is a Python object and carries additional information such as the traceback\n    of the construction of the nodes in the graph.\n\n    Args:\n      python_graph: (ops.Graph) The Python Graph object.\n    '
        self._python_graph = python_graph
        self._node_traceback = {}
        if self._python_graph:
            for op in self._python_graph.get_operations():
                self._node_traceback[op.name] = tuple(map(tuple, op.traceback))

    @property
    def python_graph(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the Python graph.\n\n    Returns:\n      If the Python graph has been set, returns a `tf.Graph` object. Otherwise,\n      returns None.\n    '
        return self._python_graph

    @property
    def core_metadata(self):
        if False:
            print('Hello World!')
        'Metadata about the `Session.run()` call from the core runtime.\n\n    Of the three counters available in the return value, `global_step` is\n    supplied by the caller of the debugged `Session.run()`, while\n    `session_run_index` and `executor_step_index` are determined by the state\n    of the core runtime, automatically. For the same fetch list, feed keys and\n    debug tensor watch options, the same executor will be used and\n    `executor_step_index` should increase by one at a time. However, runs with\n    different fetch lists, feed keys and debug_tensor watch options that all\n    share the same `Session` object can lead to gaps in `session_run_index`.\n\n    Returns:\n      If core metadata are loaded, a `namedtuple` with the fields:\n        `global_step`: A global step count supplied by the caller of\n          `Session.run()`. It is optional to the caller. If the caller did not\n          supply this parameter, its value will be -1.\n        `session_run_index`: A sorted index for Run() calls to the underlying\n          TensorFlow `Session` object.\n        `executor_step_index`: A counter for invocations of a given runtime\n          executor. The same executor is re-used for the same fetched tensors,\n          target nodes, input feed keys and debug tensor watch options.\n        `input_names`: Names of the input (feed) Tensors.\n        `output_names`: Names of the output (fetched) Tensors.\n        `target_nodes`: Names of the target nodes.\n      If the core metadata have not been loaded, `None`.\n      If more than one core metadata files exist, return a list of the\n        `nametuple` described above.\n    '
        output = self._core_metadata
        return output[0] if len(output) == 1 else output

    @property
    def dumped_tensor_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve dumped tensor data.'
        if len(self.devices()) == 1:
            return self._dump_tensor_data[self.devices()[0]]
        else:
            all_devices_data = self._dump_tensor_data.values()
            data = []
            for device_data in all_devices_data:
                data.extend(device_data)
            return sorted(data, key=lambda x: x.extended_timestamp)

    @property
    def t0(self):
        if False:
            return 10
        'Absolute timestamp of the first dumped tensor across all devices.\n\n    Returns:\n      (`int`) absolute timestamp of the first dumped tensor, in microseconds.\n    '
        return self._t0

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        'Total number of dumped tensors in the dump root directory.\n\n    Returns:\n      (`int`) The total number of dumped tensors in the dump root directory.\n    '
        return sum((len(self._dump_tensor_data[device_name]) for device_name in self._dump_tensor_data))

    def _load_partition_graphs(self, client_partition_graphs, validate):
        if False:
            for i in range(10):
                print('nop')
        'Load and process partition graphs.\n\n    Load the graphs; parse the input and control input structure; obtain the\n    device and op type of each node; remove the Copy and debug ops inserted\n    by the debugger. The gathered information can be used to validate the\n    tensor dumps.\n\n    Args:\n      client_partition_graphs: A repeated field of GraphDefs representing the\n        partition graphs executed by the TensorFlow runtime, from the Python\n        client. These partition graphs are used only if partition graphs\n        cannot be loaded from the dump directory on the file system.\n      validate: (`bool`) Whether the dump files are to be validated against the\n        partition graphs.\n\n    Raises:\n      ValueError: If the partition GraphDef of one or more devices fail to be\n        loaded.\n    '
        self._debug_graphs = {}
        self._node_devices = {}
        partition_graphs_and_device_names = []
        for device_name in self._device_names:
            partition_graph = None
            if device_name in self._dump_graph_file_paths:
                partition_graph = _load_graph_def_from_event_file(self._dump_graph_file_paths[device_name])
            else:
                logging.warn('Failed to load partition graphs for device %s from disk. As a fallback, the client graphs will be used. This may cause mismatches in device names.' % device_name)
                partition_graph = self._find_partition_graph(client_partition_graphs, device_name)
            if partition_graph:
                partition_graphs_and_device_names.append((partition_graph, device_name))
        for (partition_graph, maybe_device_name) in partition_graphs_and_device_names:
            debug_graph = debug_graphs.DebugGraph(partition_graph, device_name=maybe_device_name)
            self._debug_graphs[debug_graph.device_name] = debug_graph
            self._collect_node_devices(debug_graph)
            if validate and debug_graph.device_name in self._dump_tensor_data:
                self._validate_dump_with_graphs(debug_graph.device_name)

    def _find_partition_graph(self, partition_graphs, device_name):
        if False:
            return 10
        if partition_graphs is None:
            return None
        else:
            for graph_def in partition_graphs:
                for node_def in graph_def.node:
                    if node_def.device == device_name:
                        return graph_def
            return None

    def _collect_node_devices(self, debug_graph):
        if False:
            print('Hello World!')
        for node_name in debug_graph.node_devices:
            if node_name in self._node_devices:
                self._node_devices[node_name] = self._node_devices[node_name].union(debug_graph.node_devices[node_name])
            else:
                self._node_devices[node_name] = debug_graph.node_devices[node_name]

    def _validate_dump_with_graphs(self, device_name):
        if False:
            print('Hello World!')
        "Validate the dumped tensor data against the partition graphs.\n\n    Only the watched nodes are validated by this method, because tfdbg allows\n    clients to watch only a subset of the nodes.\n\n    Args:\n      device_name: (`str`) device name.\n\n    Raises:\n      LookupError: If the partition graphs have not been loaded yet.\n      ValueError: If dumps contain node names not found in partition graph.\n        Or if the temporal order of the dump's timestamps violate the\n        input relations on the partition graphs.\n    "
        if not self._debug_graphs:
            raise LookupError('No partition graphs loaded for device %s' % device_name)
        debug_graph = self._debug_graphs[device_name]
        for datum in self._dump_tensor_data[device_name]:
            if datum.node_name not in debug_graph.node_inputs:
                raise ValueError("Node name '%s' is not found in partition graphs of device %s." % (datum.node_name, device_name))
        pending_inputs = {}
        for node in debug_graph.node_inputs:
            pending_inputs[node] = []
            inputs = debug_graph.node_inputs[node]
            for inp in inputs:
                inp_node = debug_graphs.get_node_name(inp)
                inp_output_slot = debug_graphs.get_output_slot(inp)
                if inp_node in self._debug_watches[device_name] and inp_output_slot in self._debug_watches[device_name][inp_node] and (debug_graph.node_op_types.get(inp) not in ('Enter', 'NextIteration')) and ((inp_node, inp_output_slot) not in pending_inputs[node]):
                    pending_inputs[node].append((inp_node, inp_output_slot))
        for (i, datum) in enumerate(self._dump_tensor_data[device_name]):
            node = datum.node_name
            slot = datum.output_slot
            if not self._satisfied_at_timestamp(device_name, pending_inputs[node], datum.timestamp, start_i=i + 1):
                raise ValueError('Causality violated in timing relations of debug dumps: %s (%d): these input(s) are not satisfied: %s' % (node, datum.timestamp, repr(pending_inputs[node])))
            recipients = debug_graph.node_recipients[node]
            for recipient in recipients:
                recipient_pending_inputs = pending_inputs[recipient]
                if (node, slot) in recipient_pending_inputs:
                    if self.node_op_type(recipient) == 'Merge':
                        del recipient_pending_inputs[:]
                    else:
                        del recipient_pending_inputs[recipient_pending_inputs.index((node, slot))]

    def _satisfied_at_timestamp(self, device_name, pending, timestamp, start_i=0):
        if False:
            while True:
                i = 10
        'Determine whether pending inputs are satisfied at given timestamp.\n\n    Note: This method mutates the input argument "pending".\n\n    Args:\n      device_name: (str) device name.\n      pending: A list of 2-tuple (node_name, output_slot): the dependencies to\n        check.\n      timestamp: (int) the timestamp in question.\n      start_i: (int) the index in self._dump_tensor_data to start searching for\n        the timestamp.\n\n    Returns:\n      (bool) Whether all the dependencies in pending are satisfied at the\n        timestamp. If pending is empty to begin with, return True.\n    '
        if not pending:
            return True
        for datum in self._dump_tensor_data[device_name][start_i:]:
            if datum.timestamp > timestamp:
                break
            if datum.timestamp == timestamp and (datum.node_name, datum.output_slot) in pending:
                pending.remove((datum.node_name, datum.output_slot))
                if not pending:
                    return True
        return not pending

    def loaded_partition_graphs(self):
        if False:
            print('Hello World!')
        'Test whether partition graphs have been loaded.'
        return bool(self._debug_graphs)

    def partition_graphs(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the partition graphs.\n\n    Returns:\n      Partition graphs as a list of GraphDef.\n\n    Raises:\n      LookupError: If no partition graphs have been loaded.\n    '
        if not self._debug_graphs:
            raise LookupError('No partition graphs have been loaded.')
        return [self._debug_graphs[key].debug_graph_def for key in self._debug_graphs]

    def reconstructed_non_debug_partition_graphs(self):
        if False:
            print('Hello World!')
        'Reconstruct partition graphs with the debugger-inserted ops stripped.\n\n    The reconstructed partition graphs are identical to the original (i.e.,\n    non-debugger-decorated) partition graphs except in the following respects:\n      1) The exact names of the runtime-inserted internal nodes may differ.\n         These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.\n      2) As a consequence of 1, the nodes that receive input directly from such\n         send- and recv-type ops will have different input names.\n      3) The parallel_iteration attribute of while-loop Enter ops are set to 1.\n\n    Returns:\n      A dict mapping device names (`str`s) to reconstructed\n      `tf.compat.v1.GraphDef`s.\n    '
        non_debug_graphs = {}
        for key in self._debug_graphs:
            non_debug_graphs[key] = self._debug_graphs[key].non_debug_graph_def
        return non_debug_graphs

    @property
    def run_fetches_info(self):
        if False:
            i = 10
            return i + 15
        'Get a str representation of the fetches used in the Session.run() call.\n\n    Returns:\n      If the information is available from one `Session.run` call, a `str`\n        obtained from `repr(fetches)`.\n      If the information is available from multiple `Session.run` calls, a\n        `list` of `str` from `repr(fetches)`.\n      If the information is not available, `None`.\n    '
        output = self._run_fetches_info
        return output[0] if len(output) == 1 else output

    @property
    def run_feed_keys_info(self):
        if False:
            while True:
                i = 10
        'Get a str representation of the feed_dict used in the Session.run() call.\n\n    Returns:\n      If the information is available from one `Session.run` call, a `str`\n        obtained from `repr(feed_dict)`.\n      If the information is available from multiple `Session.run` calls, a\n        `list` of `str` obtained from `repr(feed_dict)`.\n      If the information is not available, `None`.\n    '
        output = self._run_feed_keys_info
        return output[0] if len(output) == 1 else output

    def _infer_device_name(self, device_name, node_name):
        if False:
            print('Hello World!')
        "Infer the device name given node name.\n\n    If device_name is provided (i.e., not None), it'll be simply returned right\n    away.\n\n    Args:\n      device_name: (str or None) name of the device. If None, will try to infer\n        the device name by looking at the available nodes.\n      node_name: (str) name of the node.\n\n    Returns:\n      (str) Inferred name of the device, if available.\n\n    Raises:\n      ValueError: If the node name does not exist on any of the available\n        devices or if there are multiple devices that contain the node with\n        the given name.\n    "
        if device_name is None:
            if node_name in self._node_devices:
                if len(self._node_devices[node_name]) == 1:
                    return list(self._node_devices[node_name])[0]
                else:
                    raise ValueError("There are multiple (%d) devices with nodes named '%s' but device_name is not specified." % (len(self._node_devices[node_name]), node_name))
            else:
                raise ValueError("None of the %d device(s) has a node named '%s'." % (len(self._device_names), node_name))
        else:
            return device_name

    def nodes(self, device_name=None):
        if False:
            i = 10
            return i + 15
        "Get a list of all nodes from the partition graphs.\n\n    Args:\n      device_name: (`str`) name of device. If None, all nodes from all available\n        devices will be included.\n\n    Returns:\n      All nodes' names, as a list of str.\n\n    Raises:\n      LookupError: If no partition graphs have been loaded.\n      ValueError: If specified node name does not exist.\n    "
        if not self._debug_graphs:
            raise LookupError('No partition graphs have been loaded.')
        if device_name is None:
            nodes = []
            for device_name in self._debug_graphs:
                nodes.extend(self._debug_graphs[device_name].node_inputs.keys())
            return nodes
        else:
            if device_name not in self._debug_graphs:
                raise ValueError('Invalid device name: %s' % device_name)
            return self._debug_graphs[device_name].node_inputs.keys()

    def node_attributes(self, node_name, device_name=None):
        if False:
            while True:
                i = 10
        'Get the attributes of a node.\n\n    Args:\n      node_name: Name of the node in question.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      Attributes of the node.\n\n    Raises:\n      LookupError: If no partition graphs have been loaded.\n    '
        if not self._debug_graphs:
            raise LookupError('No partition graphs have been loaded.')
        device_name = self._infer_device_name(device_name, node_name)
        return self._debug_graphs[device_name].node_attributes[node_name]

    def node_inputs(self, node_name, is_control=False, device_name=None):
        if False:
            print('Hello World!')
        'Get the inputs of given node according to partition graphs.\n\n    Args:\n      node_name: Name of the node.\n      is_control: (`bool`) Whether control inputs, rather than non-control\n        inputs, are to be returned.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      (`list` of `str`) inputs to the node, as a list of node names.\n\n    Raises:\n      LookupError: If node inputs and control inputs have not been loaded\n         from partition graphs yet.\n    '
        if not self._debug_graphs:
            raise LookupError('Node inputs are not loaded from partition graphs yet.')
        device_name = self._infer_device_name(device_name, node_name)
        if is_control:
            return self._debug_graphs[device_name].node_ctrl_inputs[node_name]
        else:
            return self._debug_graphs[device_name].node_inputs[node_name]

    def transitive_inputs(self, node_name, include_control=True, include_reversed_ref=False, device_name=None):
        if False:
            for i in range(10):
                print('nop')
        'Get the transitive inputs of given node according to partition graphs.\n\n    Args:\n      node_name: Name of the node.\n      include_control: Include control inputs (True by default).\n      include_reversed_ref: Whether a ref input, say from A to B, is to be also\n        considered as an input from B to A. The rationale is that ref inputs\n        generally let the recipient (e.g., B in this case) mutate the value of\n        the source (e.g., A in this case). So the reverse direction of the ref\n        edge reflects the direction of information flow.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      (`list` of `str`) all transitive inputs to the node, as a list of node\n        names.\n\n    Raises:\n      LookupError: If node inputs and control inputs have not been loaded\n         from partition graphs yet.\n    '
        if not self._debug_graphs:
            raise LookupError('Node inputs are not loaded from partition graphs yet.')
        device_name = self._infer_device_name(device_name, node_name)
        input_lists = [self._debug_graphs[device_name].node_inputs]
        if include_control:
            input_lists.append(self._debug_graphs[device_name].node_ctrl_inputs)
        if include_reversed_ref:
            input_lists.append(self._debug_graphs[device_name].node_reversed_ref_inputs)
        tracer = debug_graphs.DFSGraphTracer(input_lists, skip_node_names=self._get_merge_node_names(device_name))
        tracer.trace(node_name)
        return tracer.inputs()

    def _get_merge_node_names(self, device_name):
        if False:
            while True:
                i = 10
        'Lazily get a list of Merge nodes on a given device.'
        if device_name not in self._device_names:
            raise ValueError('Invalid device name: %s' % device_name)
        if not hasattr(self, '_merge_node_names'):
            self._merge_node_names = {}
        if device_name not in self._merge_node_names:
            debug_graph = self._debug_graphs[device_name]
            self._merge_node_names[device_name] = [node for node in debug_graph.node_op_types if debug_graph.node_op_types[node] == 'Merge']
        return self._merge_node_names[device_name]

    def find_some_path(self, src_node_name, dst_node_name, include_control=True, include_reversed_ref=False, device_name=None):
        if False:
            print('Hello World!')
        'Find a path between a source node and a destination node.\n\n    Limitation: the source and destination are required to be on the same\n    device, i.e., this method does not yet take into account Send/Recv nodes\n    across devices.\n\n    TODO(cais): Make this method work across device edges by tracing Send/Recv\n      nodes.\n\n    Args:\n      src_node_name: (`str`) name of the source node or name of an output tensor\n        of the node.\n      dst_node_name: (`str`) name of the destination node or name of an output\n        tensor of the node.\n      include_control: (`bool`) whrther control edges are considered in the\n        graph tracing.\n      include_reversed_ref: Whether a ref input, say from A to B, is to be also\n        considered as an input from B to A. The rationale is that ref inputs\n        generally let the recipient (e.g., B in this case) mutate the value of\n        the source (e.g., A in this case). So the reverse direction of the ref\n        edge reflects the direction of information flow.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      A path from the src_node_name to dst_node_name, as a `list` of `str`, if\n      it exists. The list includes src_node_name as the first item and\n      dst_node_name as the last.\n      If such a path does not exist, `None`.\n\n    Raises:\n      ValueError: If the source and destination nodes are not on the same\n        device.\n    '
        src_device_name = self._infer_device_name(device_name, src_node_name)
        dst_device_name = self._infer_device_name(device_name, dst_node_name)
        if src_device_name != dst_device_name:
            raise ValueError('Source (%s) and destination (%s) are not on the same device: %s vs. %s' % (src_node_name, dst_node_name, src_device_name, dst_device_name))
        input_lists = [self._debug_graphs[dst_device_name].node_inputs]
        debug_graph = self._debug_graphs[dst_device_name]
        if include_control:
            input_lists.append(debug_graph.node_ctrl_inputs)
        if include_reversed_ref:
            input_lists.append(debug_graph.node_reversed_ref_inputs)
        tracer = debug_graphs.DFSGraphTracer(input_lists, skip_node_names=self._get_merge_node_names(dst_device_name), destination_node_name=src_node_name)
        try:
            tracer.trace(dst_node_name)
        except debug_graphs.GraphTracingReachedDestination:
            inputs = [dst_node_name] + tracer.inputs()
            depth_list = [0] + tracer.depth_list()
            path = []
            curr_depth = depth_list[-1]
            for (inp, depth) in zip(reversed(inputs), reversed(depth_list)):
                if depth == curr_depth:
                    path.append(inp)
                    curr_depth -= 1
            return path

    def node_recipients(self, node_name, is_control=False, device_name=None):
        if False:
            while True:
                i = 10
        "Get recipient of the given node's output according to partition graphs.\n\n    Args:\n      node_name: (`str`) name of the node.\n      is_control: (`bool`) whether control outputs, rather than non-control\n        outputs, are to be returned.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      (`list` of `str`) all inputs to the node, as a list of node names.\n\n    Raises:\n      LookupError: If node inputs and control inputs have not been loaded\n         from partition graphs yet.\n    "
        if not self._debug_graphs:
            raise LookupError('Node recipients are not loaded from partition graphs yet.')
        device_name = self._infer_device_name(device_name, node_name)
        debug_graph = self._debug_graphs[device_name]
        if is_control:
            return debug_graph.node_ctrl_recipients[node_name]
        else:
            return debug_graph.node_recipients[node_name]

    def devices(self):
        if False:
            while True:
                i = 10
        'Get the list of device names.\n\n    Returns:\n      (`list` of `str`) names of the devices.\n    '
        return self._device_names

    def node_exists(self, node_name, device_name=None):
        if False:
            print('Hello World!')
        'Test if a node exists in the partition graphs.\n\n    Args:\n      node_name: (`str`) name of the node to be checked.\n      device_name: optional device name. If None, will search for the node\n        on all available devices. Otherwise, search for the node only on\n        the given device.\n\n    Returns:\n      A boolean indicating whether the node exists.\n\n    Raises:\n      LookupError: If no partition graphs have been loaded yet.\n      ValueError: If device_name is specified but cannot be found.\n    '
        if not self._debug_graphs:
            raise LookupError('Nodes have not been loaded from partition graphs yet.')
        if device_name is not None and device_name not in self._debug_graphs:
            raise ValueError("The specified device_name '%s' cannot be found." % device_name)
        for (_, debug_graph) in self._debug_graphs.items():
            if node_name in debug_graph.node_inputs:
                return True
        return False

    def node_device(self, node_name):
        if False:
            while True:
                i = 10
        'Get the names of the devices that has nodes of the specified name.\n\n    Args:\n      node_name: (`str`) name of the node.\n\n    Returns:\n      (`str` or `list` of `str`) name of the device(s) on which the node of the\n        given name is found. Returns a `str` if there is only one such device,\n        otherwise return a `list` of `str`.\n\n    Raises:\n      LookupError: If node inputs and control inputs have not been loaded\n         from partition graphs yet.\n      ValueError: If the node does not exist in partition graphs.\n    '
        if not self._debug_graphs:
            raise LookupError('Node devices are not loaded from partition graphs yet.')
        if node_name not in self._node_devices:
            raise ValueError("Node '%s' does not exist in partition graphs." % node_name)
        output = list(self._node_devices[node_name])
        return output[0] if len(output) == 1 else output

    def node_op_type(self, node_name, device_name=None):
        if False:
            return 10
        'Get the op type of given node.\n\n    Args:\n      node_name: (`str`) name of the node.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      (`str`) op type of the node.\n\n    Raises:\n      LookupError: If node op types have not been loaded\n         from partition graphs yet.\n    '
        if not self._debug_graphs:
            raise LookupError('Node op types are not loaded from partition graphs yet.')
        device_name = self._infer_device_name(device_name, node_name)
        return self._debug_graphs[device_name].node_op_types[node_name]

    def debug_watch_keys(self, node_name, device_name=None):
        if False:
            return 10
        'Get all tensor watch keys of given node according to partition graphs.\n\n    Args:\n      node_name: (`str`) name of the node.\n      device_name: (`str`) name of the device. If there is only one device or if\n        node_name exists on only one device, this argument is optional.\n\n    Returns:\n      (`list` of `str`) all debug tensor watch keys. Returns an empty list if\n        the node name does not correspond to any debug watch keys.\n\n    Raises:\n      `LookupError`: If debug watch information has not been loaded from\n        partition graphs yet.\n    '
        try:
            device_name = self._infer_device_name(device_name, node_name)
        except ValueError:
            return []
        if node_name not in self._debug_watches[device_name]:
            return []
        watch_keys = []
        for watched_slot in self._debug_watches[device_name][node_name]:
            debug_ops = self._debug_watches[device_name][node_name][watched_slot]
            for debug_op in debug_ops:
                watch_keys.append(_get_tensor_watch_key(node_name, watched_slot, debug_op))
        return watch_keys

    def watch_key_to_data(self, debug_watch_key, device_name=None):
        if False:
            return 10
        'Get all `DebugTensorDatum` instances corresponding to a debug watch key.\n\n    Args:\n      debug_watch_key: (`str`) debug watch key.\n      device_name: (`str`) name of the device. If there is only one device or if\n        the specified debug_watch_key exists on only one device, this argument\n        is optional.\n\n    Returns:\n      A list of `DebugTensorDatum` instances that correspond to the debug watch\n      key. If the watch key does not exist, returns an empty list.\n\n    Raises:\n      ValueError: If there are multiple devices that have the debug_watch_key,\n        but device_name is not specified.\n    '
        if device_name is None:
            matching_device_names = [name for name in self._watch_key_to_datum if debug_watch_key in self._watch_key_to_datum[name]]
            if not matching_device_names:
                return []
            elif len(matching_device_names) == 1:
                device_name = matching_device_names[0]
            else:
                raise ValueError("The debug watch key '%s' exists on multiple (%d) devices, but device name is not specified." % (debug_watch_key, len(matching_device_names)))
        elif device_name not in self._debug_key_to_datum:
            raise ValueError("There is no device named '%s' consisting of debug watch keys." % device_name)
        return self._watch_key_to_datum[device_name].get(debug_watch_key, [])

    def find(self, predicate, first_n=0, device_name=None, exclude_node_names=None):
        if False:
            print('Hello World!')
        "Find dumped tensor data by a certain predicate.\n\n    Args:\n      predicate: A callable that takes two input arguments:\n\n        ```python\n        def predicate(debug_tensor_datum, tensor):\n          # returns a bool\n        ```\n\n        where `debug_tensor_datum` is an instance of `DebugTensorDatum`, which\n        carries the metadata, such as the `Tensor`'s node name, output slot\n        timestamp, debug op name, etc.; and `tensor` is the dumped tensor value\n        as a `numpy.ndarray`.\n      first_n: (`int`) return only the first n `DebugTensotDatum` instances (in\n        time order) for which the predicate returns True. To return all the\n        `DebugTensotDatum` instances, let first_n be <= 0.\n      device_name: optional device name.\n      exclude_node_names: Optional regular expression to exclude nodes with\n        names matching the regular expression.\n\n    Returns:\n      A list of all `DebugTensorDatum` objects in this `DebugDumpDir` object\n       for which predicate returns True, sorted in ascending order of the\n       timestamp.\n    "
        if exclude_node_names:
            exclude_node_names = re.compile(exclude_node_names)
        matched_data = []
        for device in self._dump_tensor_data if device_name is None else (self._dump_tensor_data[device_name],):
            for datum in self._dump_tensor_data[device]:
                if exclude_node_names and exclude_node_names.match(datum.node_name):
                    continue
                if predicate(datum, datum.get_tensor()):
                    matched_data.append(datum)
                    if first_n > 0 and len(matched_data) >= first_n:
                        return matched_data
        return matched_data

    def get_tensor_file_paths(self, node_name, output_slot, debug_op, device_name=None):
        if False:
            return 10
        'Get the file paths from a debug-dumped tensor.\n\n    Args:\n      node_name: (`str`) name of the node that the tensor is produced by.\n      output_slot: (`int`) output slot index of tensor.\n      debug_op: (`str`) name of the debug op.\n      device_name: (`str`) name of the device. If there is only one device or if\n        the specified debug_watch_key exists on only one device, this argument\n        is optional.\n\n    Returns:\n      List of file path(s) loaded. This is a list because each debugged tensor\n        may be dumped multiple times.\n\n    Raises:\n      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor does not exist in\n        the debug-dump data.\n    '
        device_name = self._infer_device_name(device_name, node_name)
        watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
        if watch_key not in self._watch_key_to_datum[device_name]:
            raise WatchKeyDoesNotExistInDebugDumpDirError('Watch key "%s" does not exist in the debug dump of device %s' % (watch_key, device_name))
        return [datum.file_path for datum in self._watch_key_to_datum[device_name][watch_key]]

    def get_tensors(self, node_name, output_slot, debug_op, device_name=None):
        if False:
            while True:
                i = 10
        'Get the tensor value from for a debug-dumped tensor.\n\n    The tensor may be dumped multiple times in the dump root directory, so a\n    list of tensors (`numpy.ndarray`) is returned.\n\n    Args:\n      node_name: (`str`) name of the node that the tensor is produced by.\n      output_slot: (`int`) output slot index of tensor.\n      debug_op: (`str`) name of the debug op.\n      device_name: (`str`) name of the device. If there is only one device or if\n        the specified debug_watch_key exists on only one device, this argument\n        is optional.\n\n    Returns:\n      List of tensors (`numpy.ndarray`) loaded from the debug-dump file(s).\n\n    Raises:\n      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor does not exist in\n        the debug-dump data.\n    '
        watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
        try:
            device_name = self._infer_device_name(device_name, node_name)
            return [datum.get_tensor() for datum in self._watch_key_to_datum[device_name][watch_key]]
        except (ValueError, KeyError):
            raise WatchKeyDoesNotExistInDebugDumpDirError('Watch key "%s" does not exist in the debug dump of device %s' % (watch_key, device_name))

    def get_rel_timestamps(self, node_name, output_slot, debug_op, device_name=None):
        if False:
            i = 10
            return i + 15
        'Get the relative timestamp from for a debug-dumped tensor.\n\n    Relative timestamp means (absolute timestamp - `t0`), where `t0` is the\n    absolute timestamp of the first dumped tensor in the dump root. The tensor\n    may be dumped multiple times in the dump root directory, so a list of\n    relative timestamps (`numpy.ndarray`) is returned.\n\n    Args:\n      node_name: (`str`) name of the node that the tensor is produced by.\n      output_slot: (`int`) output slot index of tensor.\n      debug_op: (`str`) name of the debug op.\n      device_name: (`str`) name of the device. If there is only one device or if\n        the specified debug_watch_key exists on only one device, this argument\n        is optional.\n\n    Returns:\n      (`list` of `int`) list of relative timestamps.\n\n    Raises:\n      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor watch key does not\n        exist in the debug dump data.\n    '
        device_name = self._infer_device_name(device_name, node_name)
        watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
        if watch_key not in self._watch_key_to_datum[device_name]:
            raise WatchKeyDoesNotExistInDebugDumpDirError('Watch key "%s" does not exist in the debug dump' % watch_key)
        return self._watch_key_to_rel_time[device_name][watch_key]

    def get_dump_sizes_bytes(self, node_name, output_slot, debug_op, device_name=None):
        if False:
            return 10
        'Get the sizes of the dump files for a debug-dumped tensor.\n\n    Unit of the file size: byte.\n\n    Args:\n      node_name: (`str`) name of the node that the tensor is produced by.\n      output_slot: (`int`) output slot index of tensor.\n      debug_op: (`str`) name of the debug op.\n      device_name: (`str`) name of the device. If there is only one device or if\n        the specified debug_watch_key exists on only one device, this argument\n        is optional.\n\n    Returns:\n      (`list` of `int`): list of dump file sizes in bytes.\n\n    Raises:\n      WatchKeyDoesNotExistInDebugDumpDirError: If the tensor watch key does not\n        exist in the debug dump data.\n    '
        device_name = self._infer_device_name(device_name, node_name)
        watch_key = _get_tensor_watch_key(node_name, output_slot, debug_op)
        if watch_key not in self._watch_key_to_datum[device_name]:
            raise WatchKeyDoesNotExistInDebugDumpDirError('Watch key "%s" does not exist in the debug dump of device %s' % (watch_key, device_name))
        return self._watch_key_to_dump_size_bytes[device_name][watch_key]

    def node_traceback(self, element_name):
        if False:
            return 10
        "Try to retrieve the Python traceback of node's construction.\n\n    Args:\n      element_name: (`str`) Name of a graph element (node or tensor).\n\n    Returns:\n      (list) The traceback list object as returned by the `extract_trace`\n        method of Python's traceback module.\n\n    Raises:\n      LookupError: If Python graph is not available for traceback lookup.\n      KeyError: If the node cannot be found in the Python graph loaded.\n    "
        if self._python_graph is None:
            raise LookupError('Python graph is not available for traceback lookup')
        node_name = debug_graphs.get_node_name(element_name)
        if node_name not in self._node_traceback:
            raise KeyError('Cannot find node "%s" in Python graph' % node_name)
        return self._node_traceback[node_name]