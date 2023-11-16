"""Profiler for TensorFlow models that outputs data in pprof format.

See https://github.com/google/pprof/blob/master/proto/profile.proto for pprof
profile format.
The following needs to be set for profiler to work:
  * trace_level needs to be set to FULL_TRACE
  * run_metadata object should be passed in to session.run call

Sample usage:
  options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.compat.v1.RunMetadata()

  with tf.compat.v1.Session as sess:
    ...
    sess.run(computation, run_metadata=run_metadata, options=options)
  pprof_profiler.profile(sess.graph, run_metadata, output_dir)


  The code above would output a pprof profile to separate output_dir/.*.pb.gz
  file for each device. These files can be passed to pprof for formatting.
  For e.g.:
     pprof -png --nodecount=100 --sample_index=1 output_dir/profile_output.pb.gz
"""
from collections import defaultdict
from collections import namedtuple
import gzip
import os
import string
import sys
import time
from proto import profile_pb2
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
ProfileDatum = namedtuple('ProfileDatum', ['node_exec_stats', 'op_type', 'traceback'])

class StringTable(object):
    """Keeps track of strings to add to string_table in pprof proto."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._string_table = ['']
        self._string_to_index = {'': 0}

    def index_of(self, value_str):
        if False:
            print('Hello World!')
        'Get index of value_str in the string table.\n\n    If value_str is not in the string table, we will add it at the end\n    and then return the new index.\n    Args:\n      value_str: (string) Value to lookup/add in/to the string table.\n\n    Returns:\n      Index of value_str in the string table.\n    '
        if value_str is None:
            value_str = ''
        if value_str in self._string_to_index:
            return self._string_to_index[value_str]
        index = len(self._string_table)
        self._string_table.append(value_str)
        self._string_to_index[value_str] = index
        return index

    def next_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets index that would be assigned to the next added string.\n\n    Returns:\n      Index of the next string if it was added.\n    '
        return len(self._string_table)

    def string_table(self):
        if False:
            print('Hello World!')
        "Returns a list of strings to store in pprof's string_table."
        return self._string_table

class Functions(object):
    """Keeps track of `Function` protos for pprof profile."""

    def __init__(self, string_table):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      string_table: A `StringTable` object.\n    '
        self._string_table = string_table
        self._function_key_to_function = {}

    def index_of(self, file_path, function_name, function_start_line):
        if False:
            return 10
        'Returns index of the function, adding the function if needed.\n\n    Args:\n      file_path: (string) Path to file where the function is defined.\n      function_name: (string) Function name.\n      function_start_line: (integer) Start line number of function definition.\n\n    Returns:\n      Function index.\n    '
        function_key = (file_path, function_name, function_start_line)
        if function_key in self._function_key_to_function:
            return self._function_key_to_function[function_key].id
        else:
            function_index = len(self._function_key_to_function) + 1
            function = profile_pb2.Function()
            function.id = function_index
            function.name = self._string_table.index_of(function_name)
            function.filename = self._string_table.index_of(file_path)
            function.start_line = function_start_line
            self._function_key_to_function[function_key] = function
            return function_index

    def function_protos(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of `profile_pb2.Function` protos.'
        return self._function_key_to_function.values()

class Locations(object):
    """Keeps track of `Location` protos for pprof profile.

  `Locations` store information about function call locations.
  """

    def __init__(self, functions):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      functions: A `Functions` object.\n    '
        self._functions = functions
        self._location_key_to_location = {}

    def index_of(self, file_path, line_number, called_function_name, called_file_path, called_function_start_line):
        if False:
            return 10
        'Returns index of the location, adding the location if needed.\n\n    Args:\n      file_path: (string) Path to file that makes the call.\n      line_number: (integer) Call line number.\n      called_function_name: (string) Function name of the function called at\n        `file_path` and `line_number`.\n      called_file_path: (string) Path to file where the called function is\n        defined.\n      called_function_start_line: (integer) Start line number of called\n        function definition in `called_file_path` file.\n\n    Returns:\n      Index of location.\n    '
        location_key = (file_path, called_function_name, line_number)
        if location_key in self._location_key_to_location:
            location = self._location_key_to_location[location_key]
            return location.id
        else:
            location_index = len(self._location_key_to_location) + 1
            location = profile_pb2.Location()
            location.id = location_index
            self._location_key_to_location[location_key] = location
            line = location.line.add()
            line.function_id = self._functions.index_of(called_file_path, called_function_name, called_function_start_line)
            line.line = line_number
            return location_index

    def location_protos(self):
        if False:
            return 10
        'Returns list of `profile_pb2.Location` protos.'
        return self._location_key_to_location.values()

class Samples(object):
    """Keeps track of `Sample` protos for pprof profile.

  Samples store the following statistics in order:
  count, all_time, op_time
  """

    def __init__(self, string_table):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      string_table: A `StringTable` object.\n    '
        self._string_table = string_table
        self._node_name_to_sample = {}

    def add(self, datum, location_ids):
        if False:
            i = 10
            return i + 15
        'Adds a sample data point.\n\n    Args:\n      datum: `ProfileDatum` to add a sample for.\n      location_ids: List of numberic location ids for this\n        sample.\n    '
        node_name = datum.node_exec_stats.node_name
        if node_name in self._node_name_to_sample:
            sample = self._node_name_to_sample[node_name]
            sample.location_id.extend(location_ids)
        else:
            sample = profile_pb2.Sample()
            sample.value.extend([0, 0, 0])
            label = sample.label.add()
            label.key = self._string_table.index_of('node_name')
            label.str = self._string_table.index_of(node_name)
            label = sample.label.add()
            label.key = self._string_table.index_of('op_type')
            label.str = self._string_table.index_of(datum.op_type)
            self._node_name_to_sample[node_name] = sample
        sample.value[0] += 1
        sample.value[1] += datum.node_exec_stats.all_end_rel_micros
        sample.value[2] += datum.node_exec_stats.op_end_rel_micros - datum.node_exec_stats.op_start_rel_micros

    def get_sample_protos(self):
        if False:
            while True:
                i = 10
        'Returns list of `Sample` protos for pprof profile.'
        return self._node_name_to_sample.values()

class PprofProfiler(object):
    """Creates profiles in pprof format."""

    def __init__(self, graph, run_metadata):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      graph: A `Graph` instance.\n      run_metadata: A list of `RunMetadata` objects.\n    '
        self._graph = graph
        self._run_metadata = run_metadata
        self._string_table = StringTable()
        self._functions = Functions(self._string_table)
        self._locations = Locations(self._functions)

    def profile(self):
        if False:
            for i in range(10):
                print('nop')
        'Generates pprof profiles.\n\n    Returns:\n      Dictionary mapping from device name to proto in `profile_pb2.Profile`\n      format.\n    '
        profiles = {}
        data_generator_func = self._get_profile_data_generator()
        for (device_index, device_stats) in enumerate(self._run_metadata.step_stats.dev_stats):
            pprof_proto = self._get_pprof_proto(data_generator_func(device_stats))
            if not pprof_proto.sample:
                print('Not enough data to create profile for device %s. Did you pass RunMetadata to session.run call?' % device_stats.device)
                continue
            device_count = len(self._run_metadata.step_stats.dev_stats)
            device_description = 'Device %d of %d: %s' % (device_index + 1, device_count, device_stats.device)
            device_description_str_index = self._string_table.next_index()
            pprof_proto.string_table.append(device_description)
            pprof_proto.comment.append(device_description_str_index)
            profiles[device_stats.device] = pprof_proto
        return profiles

    def _get_pprof_proto(self, profile_datum_generator):
        if False:
            while True:
                i = 10
        'Returns profile data in pprof proto format.\n\n    Args:\n      profile_datum_generator: Generator outputting `ProfileDatum` objects.\n\n    Returns:\n      A proto in pprof format.\n    '
        pprof_profile = profile_pb2.Profile()
        samples = Samples(self._string_table)
        for datum in profile_datum_generator:
            if not datum.traceback:
                continue
            stack_frame = datum.traceback[-1]
            after_apply_op = False
            location_ids = []
            for stack_frame_index in reversed(range(len(datum.traceback) - 1)):
                prev_stack_frame = stack_frame
                stack_frame = datum.traceback[stack_frame_index]
                prev_file_path = prev_stack_frame[0]
                prev_function = prev_stack_frame[2]
                prev_function_start_line = -1
                curr_file_path = stack_frame[0]
                curr_line_number = stack_frame[1]
                if not after_apply_op:
                    if prev_function == 'apply_op':
                        after_apply_op = True
                    continue
                location_index = self._locations.index_of(curr_file_path, curr_line_number, prev_function, prev_file_path, prev_function_start_line)
                location_ids.append(location_index)
            samples.add(datum, location_ids)
        sample_type_description = 'count'
        sample_type = pprof_profile.sample_type.add()
        sample_type.type = self._string_table.index_of(sample_type_description)
        sample_type.unit = self._string_table.index_of('count')
        sample_type_description = 'all_time'
        sample_type = pprof_profile.sample_type.add()
        sample_type.type = self._string_table.index_of(sample_type_description)
        sample_type.unit = self._string_table.index_of('nanoseconds')
        sample_type_description = 'op_time'
        sample_type = pprof_profile.sample_type.add()
        sample_type.type = self._string_table.index_of(sample_type_description)
        sample_type.unit = self._string_table.index_of('nanoseconds')
        pprof_profile.string_table.extend(self._string_table.string_table())
        pprof_profile.sample.extend(samples.get_sample_protos())
        pprof_profile.function.extend(self._functions.function_protos())
        pprof_profile.location.extend(self._locations.location_protos())
        return pprof_profile

    def _get_profile_data_generator(self):
        if False:
            i = 10
            return i + 15
        'Get function that generates `ProfileDatum` objects.\n\n    Returns:\n      A function that generates `ProfileDatum` objects.\n    '
        node_to_traceback = defaultdict(list)
        node_to_op_type = defaultdict(str)
        for op in self._graph.get_operations():
            node_to_traceback[op.name] = op.traceback
            node_to_op_type[op.name] = op.type

        def profile_data_generator(device_step_stats):
            if False:
                for i in range(10):
                    print('nop')
            for node_stats in device_step_stats.node_stats:
                if node_stats.node_name == '_SOURCE' or node_stats.node_name == '_SINK':
                    continue
                yield ProfileDatum(node_stats, node_to_op_type[node_stats.node_name], node_to_traceback[node_stats.node_name])
        return profile_data_generator

def get_profiles(graph, run_metadata):
    if False:
        i = 10
        return i + 15
    'Generate profiles in pprof format.\n\n  See https://github.com/google/pprof/blob/master/proto/profile.proto\n  for pprof proto format.\n\n  Args:\n    graph: A `Graph` object.\n    run_metadata: A `RunMetadata` proto.\n\n  Returns:\n    A dictionary mapping from device name to pprof proto for that device.\n  '
    return PprofProfiler(graph, run_metadata).profile()

def profile(graph, run_metadata, output_dir=None):
    if False:
        print('Hello World!')
    'Generate profiles in pprof format.\n\n  See https://github.com/google/pprof/blob/master/proto/profile.proto\n  for pprof proto format.\n\n  Args:\n    graph: A `Graph` object.\n    run_metadata: A `RunMetadata` proto.\n    output_dir: (string) Directory to output pprof profile to.\n      Profile files for each device will be stored in compressed\n      serialized proto format. If output_dir is None, profile protos\n      will be printed to stdout instead.\n\n  Returns:\n    List of output files created by this profile call.\n    (Note: this list will be empty if output_dir is None)\n  '
    profiles = get_profiles(graph, run_metadata)
    output_file_template = None
    if output_dir:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        time_suffix = time.strftime('%Y%m%d%H%M%S')
        output_file_template = os.path.join(output_dir, '%s_' + time_suffix + '.pb.gz')
    profile_files = []
    for (device, pprof_proto) in profiles.items():
        if output_file_template is None:
            print('No output directory specified, printing to stdout instead.')
            print(pprof_proto)
        else:
            device_name = str(device).strip('/').translate(maketrans('/:', '__'))
            profile_file = output_file_template % device_name
            profile_files.append(profile_file)
            with gzip.open(profile_file, 'w') as output_file:
                print('Writing profile to %s...' % profile_file)
                output_file.write(pprof_proto.SerializeToString())
    return profile_files