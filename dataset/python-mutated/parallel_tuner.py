import copy
import hashlib
import itertools
import math
import time
from collections import defaultdict
import numpy as np
from ...process_mesh import ProcessMesh
from ..completion import Completer
from ..cost import CostEstimator
from ..dist_context import _node_id
from ..dist_op import DistributedOperator
from ..operators.common import find_compatible_distributed_operator_impls
from ..parallelizer_v2 import Parallelizer
from .trial import Trial, TrialStatus
from .tunable_space import TunableSpace
from .tunable_variable import Boolean, IntRange

class ParallelTuner:

    def __init__(self, dist_context, mode='train', max_trials=25, tuner_id=None, seed=None, logger=None, loop_count=10):
        if False:
            while True:
                i = 10
        self._loop_count = loop_count
        self._estimator = None
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._mode = mode
        self._cluster = self._dist_context.cluster
        self._num_machines = self._cluster.get_num_machines()
        self._num_devices_per_machine = self._cluster.get_num_devices_per_machine()
        self._space = TunableSpace()
        self._objective = 'time'
        self._direction = 'min'
        self._max_trials = max_trials
        self._tuner_id = tuner_id
        self._seed = seed if seed is not None else 9999
        print('seed', self._seed, 'mode', self._mode, 'num_machies', self._num_machines, 'num_devices_per_machine', self._num_devices_per_machine, flush=True)
        self._seed_state = self._seed
        self._logger = logger
        self._max_collisions = 3
        self._tried_values = set()
        self._num_trials = 0
        self._rng = np.random.default_rng(self._seed)
        self._exclude_op_types = []
        self._include_op_types = []
        self._concerned_dist_ops = {}
        self._op_id_to_dist_attr_candidates = defaultdict(list)
        self._cached_dims_mapping_candidates = {}
        self._cached_candidates_info = defaultdict(list)
        self._special_ops = ['create_py_reader', 'create_double_buffer_reader', 'read', 'while', 'read_from_array', 'write_to_array']
        self._init_parallel_strategy = [None, None, None]
        self._best_parallel_strategy = [None, None, None]
        self._completer = Completer(self._dist_context)
        self._parallelizer = Parallelizer(self._mode, self._completer, self._dist_context)

    def _generate_combination(self, elements, target, idx, partial_candidate, candidates, num_candidates=None):
        if False:
            for i in range(10):
                print('nop')
        if target == 0:
            candidates.append(copy.deepcopy(partial_candidate))
            return
        if target < 0 or idx == len(elements) or len(candidates) > num_candidates:
            return
        partial_candidate.append(elements[idx])
        self._generate_combination(elements, target - elements[idx], idx, partial_candidate, candidates, num_candidates)
        partial_candidate.pop()
        self._generate_combination(elements, target, idx + 1, partial_candidate, candidates, num_candidates)

    def _permute_combination(self, combination, target, check, partial_candidate, candidates, num_candidates=None, skip_prob=None):
        if False:
            return 10
        if num_candidates is not None and len(candidates) == num_candidates:
            return
        if len(partial_candidate) == len(combination):
            candidates.append(partial_candidate)
            return
        for i in range(len(combination)):
            if check[i] == 1:
                continue
            if self._rng.choice([True, False], p=[skip_prob, 1 - skip_prob]):
                continue
            if i > 0 and combination[i] == combination[i - 1] and (check[i - 1] == 0):
                continue
            check[i] = 1
            self._permute_combination(combination, target, check, partial_candidate + [combination[i]], candidates, num_candidates, skip_prob)
            check[i] = 0

    def _partition_number(self, target):
        if False:
            for i in range(10):
                print('nop')
        log2_target = int(math.log2(target))
        elements = [pow(2, i) for i in range(log2_target)]
        if pow(2, log2_target) == target:
            elements.append(target)
        seed_candidates = []
        num_seed_candidates = 1000
        partial_results = []
        self._generate_combination(elements, target, 0, partial_results, seed_candidates, num_seed_candidates)
        candidates = []
        for seed_candidate in seed_candidates:
            cur_candidates = []
            num_cur_candidates = 16
            seed_candidate.sort()
            check = [0 for i in range(len(seed_candidate))]
            if target <= 8:
                skip_prob = 0.0
            else:
                skip_prob = len(seed_candidate) / target
            self._permute_combination(seed_candidate, target, check, [], cur_candidates, num_cur_candidates, skip_prob)
            candidates.extend(cur_candidates)
        return candidates

    def _partition_devices(self, num_machines, num_devices_per_machine):
        if False:
            for i in range(10):
                print('nop')
        inter_node_partitions = self._partition_number(num_machines)
        intra_node_partitions = self._partition_number(num_devices_per_machine)
        return (inter_node_partitions, intra_node_partitions)

    def _generate_process_mesh_list(self, inter_node_partition, intra_node_partition):
        if False:
            return 10
        process_mesh_list = []
        start_row = 0
        start_col = 0
        for m in inter_node_partition:
            start_col = 0
            for n in intra_node_partition:
                process_mesh = []
                for p in range(m):
                    start = (start_row + p) * self._num_devices_per_machine + start_col
                    tmp = []
                    for q in range(n):
                        tmp.append(start + q)
                    process_mesh.append(tmp)
                process_mesh_list.append(copy.deepcopy(process_mesh))
                start_col += n
            start_row += m
        return process_mesh_list

    def _generate_dims_mapping_candidates_helper(self, dims_mapping, dims_list, start, visited, candidates):
        if False:
            i = 10
            return i + 15
        if start == len(dims_mapping) or all(visited):
            candidates.append(copy.deepcopy(dims_mapping))
            return
        for (idx, dim) in enumerate(dims_list):
            if not visited[idx]:
                dims_mapping[start] = dim
                visited[idx] = True
                self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list, start + 1, visited, candidates)
                visited[idx] = False
        dims_mapping[start] = -1
        self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list, start + 1, visited, candidates)

    def _generate_dims_mapping_candidates(self, dims_mapping_len, process_mesh_len):
        if False:
            i = 10
            return i + 15
        assert dims_mapping_len >= 1 and process_mesh_len >= 1
        key = (dims_mapping_len, process_mesh_len)
        if key in self._cached_dims_mapping_candidates:
            return self._cached_dims_mapping_candidates[key]
        candidates = []
        dims_mapping = [-1 for i in range(dims_mapping_len)]
        dims_list = list(range(process_mesh_len))
        visited = [False for i in range(process_mesh_len)]
        self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list, 0, visited, candidates)
        self._cached_dims_mapping_candidates[key] = candidates
        return candidates

    def _generate_dist_attr_candidates(self, op_id, dist_op):
        if False:
            i = 10
            return i + 15
        process_mesh_len = 2
        serial_op = dist_op.serial_op
        op_dist_attr = dist_op.dist_attr
        if serial_op.type in self._special_ops:
            return [copy.deepcopy(op_dist_attr)]
        key = []
        key.append(serial_op.type)
        for input_name in serial_op.input_names:
            key.append(input_name)
            for input_arg_name in serial_op.input(input_name):
                key.append(len(op_dist_attr.get_input_dims_mapping(input_arg_name)))
        for output_name in serial_op.output_names:
            key.append(output_name)
            for output_arg_name in serial_op.output(output_name):
                key.append(len(op_dist_attr.get_output_dims_mapping(output_arg_name)))
        key = tuple(key)
        if key in self._cached_candidates_info:
            cached_dist_attr_candidates = []
            cached_input_arg_names = self._cached_candidates_info[key][0]
            cached_output_arg_names = self._cached_candidates_info[key][1]
            for cached_dist_attr in self._cached_candidates_info[key][2]:
                new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
                i = 0
                for input_name in serial_op.input_names:
                    for input_arg_name in serial_op.input(input_name):
                        cached_dims_mapping = cached_dist_attr.get_input_dims_mapping(cached_input_arg_names[i])
                        new_op_dist_attr.set_input_dims_mapping(input_arg_name, cached_dims_mapping)
                        i += 1
                i = 0
                for output_name in serial_op.output_names:
                    for output_arg_name in serial_op.output(output_name):
                        cached_dims_mapping = cached_dist_attr.get_output_dims_mapping(cached_output_arg_names[i])
                        new_op_dist_attr.set_output_dims_mapping(output_arg_name, cached_dims_mapping)
                        i += 1
                cached_dist_attr_candidates.append(new_op_dist_attr)
            return cached_dist_attr_candidates
        input_arg_names = []
        for input_name in serial_op.input_names:
            for input_arg_name in serial_op.input(input_name):
                input_arg_names.append(input_arg_name)
        self._cached_candidates_info[key].append(input_arg_names)
        output_arg_names = []
        for output_name in serial_op.output_names:
            for output_arg_name in serial_op.output(output_name):
                output_arg_names.append(output_arg_name)
        self._cached_candidates_info[key].append(output_arg_names)
        new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
        input_names = []
        dims_mapping_generated = []
        inputs_dist_attrs = op_dist_attr.inputs_dist_attrs
        for (tensor_name, tensor_dist_attr) in inputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            input_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append([copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(self._generate_dims_mapping_candidates(dims_mapping_len, process_mesh_len))
        input_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(input_names)
            for (i, dims_mapping) in enumerate(dims_mapping_list):
                new_op_dist_attr.set_input_dims_mapping(input_names[i], dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op, new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(new_dist_op, fwd=True)
            if dist_op_impls is not None:
                input_dims_mapping_candidates.append(dims_mapping_list)
        output_names = []
        dims_mapping_generated = []
        outputs_dist_attrs = op_dist_attr.outputs_dist_attrs
        for (tensor_name, tensor_dist_attr) in outputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            output_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append([copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(self._generate_dims_mapping_candidates(dims_mapping_len, process_mesh_len))
        output_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(output_names)
            for (i, dims_mapping) in enumerate(dims_mapping_list):
                new_op_dist_attr.set_output_dims_mapping(output_names[i], dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op, new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(new_dist_op, fwd=False)
            if dist_op_impls is not None:
                output_dims_mapping_candidates.append(dims_mapping_list)
        if not input_dims_mapping_candidates and output_dims_mapping_candidates:
            inout_dims_mapping_generated = [[[[-2]]], output_dims_mapping_candidates]
        elif input_dims_mapping_candidates and (not output_dims_mapping_candidates):
            inout_dims_mapping_generated = [input_dims_mapping_candidates, [[[-2]]]]
        elif not input_dims_mapping_candidates and (not output_dims_mapping_candidates):
            inout_dims_mapping_generated = [[[[-2]]], [[[-2]]]]
        else:
            inout_dims_mapping_generated = [input_dims_mapping_candidates, output_dims_mapping_candidates]
        cached_dist_attr_candidates = []
        for inout_dims_mapping_list in itertools.product(*inout_dims_mapping_generated):
            assert len(inout_dims_mapping_list) == 2
            if input_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[0]) == len(input_names)
            if output_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[1]) == len(output_names)
            for (i, dims_mapping) in enumerate(inout_dims_mapping_list[0]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_input_dims_mapping(input_names[i], dims_mapping)
            for (i, dims_mapping) in enumerate(inout_dims_mapping_list[1]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_output_dims_mapping(output_names[i], dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op, new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(new_dist_op, partial=False)
            if dist_op_impls is None:
                continue
            for dist_op_impl in dist_op_impls:
                new_op_dist_attr.impl_type = dist_op_impl.type
                new_op_dist_attr.impl_idx = dist_op_impl.idx
                cached_dist_attr_candidates.append(copy.deepcopy(new_op_dist_attr))
        self._cached_candidates_info[key].append(cached_dist_attr_candidates)
        return self._cached_candidates_info[key][2]

    def construct_space(self):
        if False:
            i = 10
            return i + 15
        (inter_node_partitions, intra_node_partitions) = self._partition_devices(self._num_machines, self._num_devices_per_machine)
        self._space.choice('inter_node_partitions', inter_node_partitions, default=inter_node_partitions[0])
        self._space.choice('intra_node_partitions', intra_node_partitions, default=intra_node_partitions[0])
        dist_ops = self._dist_context._dist_ops_for_program
        for (op_id, dist_op) in dist_ops.items():
            op_type = dist_op.serial_op.type
            if self._include_op_types:
                if op_type in self._include_op_types:
                    self._concerned_dist_ops[op_id] = dist_op
            else:
                self._concerned_dist_ops[op_id] = dist_op
        for (op_id, dist_op) in self._concerned_dist_ops.items():
            op_type = dist_op.serial_op.type
            if op_type in self._exclude_op_types:
                del self._concerned_dist_ops[op_id]
        print('Number of the concerned dist ops', len(self._concerned_dist_ops), flush=True)
        search_space = 1
        for (op_id, dist_op) in self._concerned_dist_ops.items():
            op_dist_attr_candidates = self._generate_dist_attr_candidates(op_id, dist_op)
            search_space *= len(op_dist_attr_candidates)
            self._space.choice(str(op_id), op_dist_attr_candidates, default=op_dist_attr_candidates[0])

    def _compute_values_hash(self, values):
        if False:
            for i in range(10):
                print('nop')
        keys = sorted(values.keys())
        s = ''.join((str(k) + '=' + str(values[k]) for k in keys))
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def _random_values(self):
        if False:
            return 10
        space = TunableSpace()
        collisions = 0
        while True:
            for v in self._space.variables.values():
                space._register(v)
                space.values[v.name] = v.random(self._seed_state)
                self._seed_state += 1
            values = space.values
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_values:
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            self._tried_values.add(values_hash)
            break
        return values

    def _populate_space(self):
        if False:
            i = 10
            return i + 15
        values = self._random_values()
        if values is None:
            return {'status': TrialStatus.STOPPED, 'values': None}
        return {'status': TrialStatus.RUNNING, 'values': values}

    def _create_trial(self):
        if False:
            print('Hello World!')
        trial_id = f'{{:0{len(str(self._max_trials))}d}}'
        trial_id = trial_id.format(self._num_trials)
        if self._max_trials and self._num_trials >= self._max_trials:
            status = TrialStatus.STOPPED
            values = None
        else:
            results = self._populate_space()
            status = results['status']
            values = results['values']
        space = TunableSpace()
        space.variables = self._space.variables
        space.values = values
        trial = Trial(tunable_space=space, trial_id=trial_id, status=status)
        self._num_trials += 1
        return trial

    def _generate_pipeline_starts(self, process_mesh_list):
        if False:
            print('Hello World!')
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stage = total_ops // total_stages
        if ops_per_stage == 0:
            return None
        pipeline_starts = []
        start = 0
        pipeline_starts.append(0)
        for _ in process_mesh_list:
            start += ops_per_stage
            pipeline_starts.append(start)
        pipeline_starts[-1] = total_ops
        directions = []
        sizes = []
        half_ops_per_stage = ops_per_stage // 2
        if half_ops_per_stage > 0 and total_stages > 1:
            new_pipeline_starts = []
            new_pipeline_starts.append(0)
            for _ in pipeline_starts[1:-1]:
                directions.append(Boolean('direction'))
                sizes.append(IntRange('size', start=0, stop=half_ops_per_stage, endpoint=True))
            for (i, start) in enumerate(pipeline_starts[1:-1]):
                direction = directions[i].random(self._seed)
                size = sizes[i].random(self._seed)
                if direction:
                    new_start = start - (size - 1)
                else:
                    new_start = start + size
                new_pipeline_starts.append(new_start)
            new_pipeline_starts.append(pipeline_starts[-1])
            print('Adjusted pipeline starts', new_pipeline_starts, half_ops_per_stage, pipeline_starts, flush=True)
            for (i, new_start) in enumerate(new_pipeline_starts[1:]):
                assert new_start > new_pipeline_starts[i]
            return new_pipeline_starts
        else:
            print('Non-adjusted pipeline starts', pipeline_starts, half_ops_per_stage, flush=True)
            return pipeline_starts

    def _apply_pipeline_partition(self, process_mesh_list):
        if False:
            for i in range(10):
                print('nop')
        op_id_to_process_mesh = {}
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stage = total_ops // total_stages
        if ops_per_stage == 0:
            return None
        pipeline_starts = self._generate_pipeline_starts(process_mesh_list)
        start_idx = 1
        sorted_op_ids = sorted(self._dist_context._dist_ops_for_program.keys())
        for (idx, op_id) in enumerate(sorted_op_ids):
            if idx < pipeline_starts[start_idx]:
                op_id_to_process_mesh[op_id] = process_mesh_list[start_idx - 1]
            else:
                start_idx += 1
                op_id_to_process_mesh[op_id] = process_mesh_list[start_idx - 1]
        return op_id_to_process_mesh

    def _amend_dist_attr(self):
        if False:
            return 10
        for dist_op in self._dist_context._dist_ops_for_program.values():
            dist_attr = dist_op.dist_attr
            process_mesh = dist_attr.process_mesh
            if process_mesh is None:
                continue
            assert process_mesh.ndim == 2
            dim_of_one = None
            dim_of_other = None
            if process_mesh.shape[0] == 1:
                dim_of_one = 0
                dim_of_other = 1
            elif process_mesh.shape[1] == 1:
                dim_of_one = 1
                dim_of_other = 0
            if dim_of_one is not None:
                dist_attr.process_mesh = ProcessMesh(process_mesh.process_ids)
                self._dist_context.add_process_mesh(dist_attr.process_mesh)
            for arg_name in dist_attr.inputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                process_mesh = dist_attr.process_mesh
                process_shape = process_mesh.shape
                tensor = dist_op.get_serial_input(arg_name)
                if dims_mapping:
                    tensor_shape = tensor.shape
                else:
                    continue
                for (i, dim_mapping) in enumerate(dims_mapping):
                    if dim_mapping != -1 and tensor_shape[i] % process_shape[dim_mapping] != 0:
                        dims_mapping[i] = -1
                    if dim_mapping != -1 and process_shape[dim_mapping] == 1:
                        dims_mapping[i] = -1
            for arg_name in dist_attr.outputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_output_dims_mapping(arg_name, new_dims_mapping)
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                process_mesh = dist_attr.process_mesh
                process_shape = process_mesh.shape
                tensor = dist_op.get_serial_output(arg_name)
                if dims_mapping:
                    tensor_shape = tensor.shape
                else:
                    continue
                for (i, dim_mapping) in enumerate(dims_mapping):
                    if dim_mapping != -1 and tensor_shape[i] % process_shape[dim_mapping] != 0:
                        dims_mapping[i] = -1
                    if dim_mapping != -1 and process_shape[dim_mapping] == 1:
                        dims_mapping[i] = -1
            dist_op_impls = find_compatible_distributed_operator_impls(dist_op, partial=False)
            serial_op_type = dist_op.serial_op.type
            if dist_op_impls is not None and (serial_op_type != 'fused_softmax_mask_upper_triangle' or self._check_fused_softmax_mask_upper_triangle(dist_op)):
                dist_op.dist_attr.impl_type = dist_op_impls[0].type
                dist_op.dist_attr.impl_idx = dist_op_impls[0].idx
            else:
                for arg_name in dist_attr.inputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                    for (i, _) in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                for arg_name in dist_attr.outputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                    for (i, _) in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                dist_op.dist_attr.impl_type = 'default'
                dist_op.dist_attr.impl_idx = 0

    def _check_fused_softmax_mask_upper_triangle(self, dist_op):
        if False:
            for i in range(10):
                print('nop')
        'The last_but_one dim should be equal to last dim.'
        input_name = dist_op.serial_op.input_arg_names[0]
        input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(input_name)
        topology = dist_op.dist_attr.process_mesh.shape
        input_tensor = dist_op.get_serial_input(input_name)
        last_but_one_dim = input_tensor.shape[-2] // topology[input_dims_mapping[-2]] if input_dims_mapping[-2] != -1 else input_tensor.shape[-2]
        last_dim = input_tensor.shape[-1] // topology[input_dims_mapping[-1]] if input_dims_mapping[-1] != -1 else input_tensor.shape[-1]
        if last_but_one_dim == last_dim:
            return True
        return False

    def _eval_trial(self, trial):
        if False:
            return 10
        if self._num_trials == 0:
            num_prev_trials = 0
        else:
            num_prev_trials = self._num_trials - 1
        results = None
        start_time = time.time()
        inter_node_partition = trial.space.values['inter_node_partitions']
        intra_node_partition = trial.space.values['intra_node_partitions']
        process_mesh_list = self._generate_process_mesh_list(inter_node_partition, intra_node_partition)
        print('\tprocess_mesh list', process_mesh_list, flush=True)
        op_id_to_process_mesh = self._apply_pipeline_partition(process_mesh_list)
        if op_id_to_process_mesh is None:
            print('Operators are less than pipeline stages', flush=True)
            return results
        op_id_to_dist_attr = {}
        for (name, value) in trial.space.values.items():
            if name != 'inter_node_partitions' and name != 'intra_node_partitions':
                op_id_to_dist_attr[int(name)] = value
        end_time = time.time()
        cur_sample_time = end_time - start_time
        self._sample_time = (num_prev_trials * self._sample_time + cur_sample_time) / self._num_trials
        print('\tsample_time', num_prev_trials, self._num_trials, self._sample_time, cur_sample_time, flush=True)
        assert len(op_id_to_process_mesh) == len(op_id_to_dist_attr)
        start_time = time.time()
        for (op_id, process_mesh) in op_id_to_process_mesh.items():
            dist_op = self._dist_context._dist_ops_for_program[op_id]
            dist_op.dist_attr = copy.deepcopy(op_id_to_dist_attr[op_id])
            assert dist_op.dist_attr.impl_type == op_id_to_dist_attr[op_id].impl_type
            assert dist_op.dist_attr.impl_idx == op_id_to_dist_attr[op_id].impl_idx
            dist_op.dist_attr.process_mesh = ProcessMesh(process_mesh)
        self._amend_dist_attr()
        self._completer._complete_tensor_dist_attr_by_op()
        self._dist_context.block_state.parse_forward_blocks(self._dist_context.serial_main_program)
        end_time = time.time()
        cur_complete_time = end_time - start_time
        self._complete_time = (num_prev_trials * self._complete_time + cur_complete_time) / self._num_trials
        print('\tcomplete_time', num_prev_trials, self._num_trials, self._complete_time, cur_complete_time, flush=True)
        start_time = time.time()
        estimate_time = self._estimate_trial()
        end_time = time.time()
        cur_estimate_time = end_time - start_time
        self._estimate_time = (num_prev_trials * self._estimate_time + cur_estimate_time) / self._num_trials
        print('\testimate_time', num_prev_trials, self._num_trials, self._estimate_time, cur_estimate_time, estimate_time, flush=True)
        results = {'estimate_time': estimate_time}
        return results

    def _update_trail(self, trial, metrics, step=0):
        if False:
            i = 10
            return i + 15
        for (metric_name, metric_value) in metrics.items():
            trial.recorder.update(metric_name, metric_value, step=step)
        return trial.status

    def _estimate_trial(self):
        if False:
            i = 10
            return i + 15
        assert self._cluster is not None
        if self._mode == 'eval':
            self._estimator = CostEstimator(self._dist_context.serial_main_program, self._cluster, loop_count=self._loop_count)
        elif self._mode == 'predict':
            self._estimator = CostEstimator(self._dist_context.serial_main_program, self._cluster, loop_count=self._loop_count)
        elif self._mode == 'train':
            serial_main_program = self._dist_context.serial_main_program
            serial_startup_program = self._dist_context.serial_startup_program
            serial_optimizer = self._dist_context.serial_optimizer
            serial_loss = self._dist_context.serial_fetch_vars['loss'][0]
            params_grads = self._parallelizer._generate_backward(serial_main_program, serial_startup_program, serial_loss)
            optimizer_ops = self._parallelizer._generate_optimizer(serial_main_program, serial_startup_program, serial_optimizer, params_grads)
            self._estimator = CostEstimator(serial_main_program, self._cluster, loop_count=self._loop_count)
        max_memory = self._estimator._estimate_max_memory_by_dist_op(self._dist_context)
        print('\tmax_memory', f'{max_memory:,}', flush=True)
        if max_memory > 32 * 0.8 * 1024 * 1024 * 1024:
            return math.inf
        else:
            global_cost = self._estimator.estimate(self._dist_context)
            return global_cost.time

    def _store_init_parallel_strategy(self):
        if False:
            print('Hello World!')
        if not self._dist_context.has_annotation or not self._dist_context.process_meshes:
            ranks = self._num_machines * self._num_devices_per_machine
            tensor_node = self._dist_context._serial_ordered_tensor_nodes[0]
            tensor_node_id = _node_id(tensor_node)
            tensor = self._dist_context._dist_tensors_for_graph[tensor_node_id].serial_tensor
            tensor_dist_attr = self._dist_context._dist_tensors_for_graph[tensor_node_id].dist_attr
            tensor_dist_attr.process_mesh = ProcessMesh(list(range(ranks)))
            self._dist_context._process_meshes.append(tensor_dist_attr.process_mesh)
            tensor_dist_attr.dims_mapping = [0] + [-1 for _ in range(len(tensor.shape) - 1)]
            tensor_dist_attr.mark_annotated('process_mesh')
            tensor_dist_attr.mark_annotated('dims_mapping')
            print('Use dp as the init parallel strategy!', flush=True)
        self._completer.complete_forward_annotation()
        self._dist_context.block_state.parse_forward_blocks(self._dist_context.serial_main_program)
        self._init_parallel_strategy[0] = copy.deepcopy(self._dist_context._dist_tensors_for_program)
        self._init_parallel_strategy[1] = copy.deepcopy(self._dist_context._dist_ops_for_program)
        self._init_parallel_strategy[2] = copy.deepcopy(self._dist_context.process_meshes)
        self._best_parallel_strategy[0] = copy.deepcopy(self._dist_context._dist_tensors_for_program)
        self._best_parallel_strategy[1] = copy.deepcopy(self._dist_context._dist_ops_for_program)
        self._best_parallel_strategy[2] = copy.deepcopy(self._dist_context._process_meshes)

    def _store_best_parallel_strategy(self):
        if False:
            while True:
                i = 10
        tmp = [None, None, None]
        tmp[0] = self._best_parallel_strategy[0]
        tmp[1] = self._best_parallel_strategy[1]
        tmp[2] = self._best_parallel_strategy[2]
        self._best_parallel_strategy[0] = self._dist_context._dist_tensors_for_program
        self._best_parallel_strategy[1] = self._dist_context._dist_ops_for_program
        self._best_parallel_strategy[2] = self._dist_context._process_meshes
        self._dist_context._dist_tensors_for_program = tmp[0]
        self._dist_context._dist_ops_for_program = tmp[1]
        self._dist_context._process_meshes = tmp[2]

    def tune(self):
        if False:
            for i in range(10):
                print('nop')
        global_start_time = time.time()
        self._dist_context._backup(serial=True, dist=True)
        self._store_init_parallel_strategy()
        init_time = self._estimate_trial()
        self._dist_context._restore(serial=True, serial_mode='to_backup', dist=True, dist_mode='to_default')
        best_time = init_time
        start_time = time.time()
        self.construct_space()
        end_time = time.time()
        print('construct_space time', self._num_trials, end_time - start_time, flush=True)
        create_trial_time = 0.0
        eval_trial_time = 0.0
        self._sample_time = 0.0
        self._complete_time = 0.0
        self._estimate_time = 0.0
        while True:
            start_time = time.time()
            trial = self._create_trial()
            if self._num_trials == 0:
                num_prev_trials = 0
            else:
                num_prev_trials = self._num_trials - 1
            end_time = time.time()
            cur_create_trial_time = end_time - start_time
            create_trial_time = (num_prev_trials * create_trial_time + cur_create_trial_time) / self._num_trials
            print('create_trial time', num_prev_trials, self._num_trials, create_trial_time, cur_create_trial_time, flush=True)
            if trial.status == TrialStatus.STOPPED:
                break
            self._dist_context._backup(serial=True, dist=False)
            start_time = time.time()
            results = self._eval_trial(trial)
            end_time = time.time()
            cur_eval_trial_time = end_time - start_time
            eval_trial_time = (num_prev_trials * eval_trial_time + cur_eval_trial_time) / self._num_trials
            print('eval_trial time', num_prev_trials, self._num_trials, eval_trial_time, cur_eval_trial_time, '\n', flush=True)
            cur_time = results['estimate_time']
            if cur_time < best_time:
                self._update_trail(trial, results)
                self._store_best_parallel_strategy()
                best_time = cur_time
            self._dist_context._restore(serial=True, serial_mode='to_backup', dist=True, dist_mode='to_default')
        self._dist_context._dist_tensors_for_program = self._best_parallel_strategy[0]
        self._dist_context._dist_ops_for_program = self._best_parallel_strategy[1]
        self._dist_context._process_meshes = self._best_parallel_strategy[2]