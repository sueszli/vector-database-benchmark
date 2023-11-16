import logging
import os
from paddle.base import core
from paddle.distributed.auto_parallel.static.cost import calc_time_by_cost_model
from ..utils.log_utils import get_logger
from .pass_base import PassContext, new_pass, register_pass
from .pass_utils import AutoParallelStreamType, _add_event_dependency, _program_for_fthenb_and_1f1b, split_program
from .pipeline_pass_base import PipelinePassBase
__not_shape_var_type__ = [core.VarDesc.VarType.READER, core.VarDesc.VarType.STEP_SCOPES, core.VarDesc.VarType.LOD_TENSOR_ARRAY, core.VarDesc.VarType.FEED_MINIBATCH, core.VarDesc.VarType.FETCH_LIST]
FORWARD = 'forward'
BACKWARD = 'backward'
OPT = 'optimizer'
logger = get_logger(logging.INFO)

@register_pass('pipeline_scheduler_FThenB')
class PipelineFThenBPass(PipelinePassBase):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def _create_job_list(self):
        if False:
            for i in range(10):
                print('nop')
        num_micro_batches = self.get_attr('num_micro_batches')
        job_list = []
        for i in range(num_micro_batches):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(i)
            job_list.append(forward_job)
        for i in range(num_micro_batches):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(i)
            job_list.append(backward_job)
        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        if False:
            i = 10
            return i + 15
        enable_send_recv_overlap = self.get_attr('enable_send_recv_overlap')
        types = [FORWARD, BACKWARD, OPT]
        sub_program_list = _program_for_fthenb_and_1f1b(program, enable_send_recv_overlap)
        return (types, sub_program_list)

@register_pass('pipeline_scheduler_1F1B')
class Pipeline1F1BPass(PipelinePassBase):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.jobs_in_stable_phase = [BACKWARD, FORWARD]
        self.set_attr('enable_backward_forward_overlap', 0)

    def _backward_forward_overlap(self, backward_program, forward_program):
        if False:
            i = 10
            return i + 15
        logger.info('Backward forward overlap enabled in 1F1B.')
        (backward_ops, forward_ops) = (backward_program.global_block().ops, forward_program.global_block().ops)
        (num_backward_ops, num_forward_ops) = (len(backward_ops), len(forward_ops))
        (backward_split_points, forward_split_points) = ([], [])
        (backward_op_id, forward_op_id) = (0, 0)
        while backward_op_id < num_backward_ops and forward_op_id < num_forward_ops:
            while backward_op_id < num_backward_ops and (not self.is_comm_op_valid_to_overlap(backward_ops[backward_op_id])):
                backward_op_id += 1
            if backward_op_id >= num_backward_ops:
                break
            backward_op_to_overlap = backward_ops[backward_op_id]
            backward_cost_to_overlap = 400
            backward_op_id += 1
            forward_op_to_overlap = forward_ops[forward_op_id]
            forward_cost_to_overlap = self._op_cost(forward_op_to_overlap)
            '\n            # Debug messages:\n            logger.info(\n                f"backward_op_to_overlap : {backward_op_to_overlap}, cost = {backward_cost_to_overlap}"\n            )\n            logger.info(\n                f"forward_op_to_overlap : {forward_op_to_overlap}, cost = {forward_cost_to_overlap}"\n            )\n            '
            while forward_op_id < num_forward_ops and backward_cost_to_overlap >= forward_cost_to_overlap:
                forward_op_id += 1
                forward_op_to_overlap = forward_ops[forward_op_id]
                forward_cost_to_overlap += self._op_cost(forward_op_to_overlap)
                '\n                # Debug messages:\n                logger.info(\n                    f"forward_op_to_overlap : {forward_op_to_overlap}, cost = {self._op_cost(forward_op_to_overlap)}"\n                )\n                '
                if self.is_comm_op_valid_to_overlap(forward_ops[forward_op_id - 1]):
                    break
            if not forward_split_points or forward_op_id > forward_split_points[-1]:
                backward_split_points.append(backward_op_id)
                forward_split_points.append(forward_op_id)
        (splitted_backward_job_types, splitted_backward_programs) = self._split_program_for_overlapping(BACKWARD, backward_program, backward_split_points)
        (splitted_forward_job_types, splitted_forward_programs) = self._split_program_for_overlapping(FORWARD, forward_program, forward_split_points)
        self._multistreaming_for_overlapping(splitted_backward_programs, BACKWARD)
        self._multistreaming_for_overlapping(splitted_forward_programs, FORWARD)
        self.jobs_in_stable_phase.clear()
        (num_splitted_backward_jobs, num_splitted_forward_jobs) = (len(splitted_backward_job_types), len(splitted_forward_job_types))
        for idx in range(max(num_splitted_backward_jobs, num_splitted_forward_jobs)):
            if idx < num_splitted_backward_jobs:
                self.jobs_in_stable_phase.append(splitted_backward_job_types[idx])
            if idx < num_splitted_forward_jobs:
                self.jobs_in_stable_phase.append(splitted_forward_job_types[idx])
        return (splitted_backward_job_types, splitted_backward_programs, splitted_forward_job_types, splitted_forward_programs)

    def _create_job_list(self):
        if False:
            for i in range(10):
                print('nop')
        num_micro_batches = self.get_attr('num_micro_batches')
        pp_stage = self.get_attr('pp_stage')
        pp_degree = self.get_attr('pp_degree')
        job_list = []
        assert pp_degree <= num_micro_batches, 'Num of micro batches should larger than or equal to pp degree.'
        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup
        forward_micro_batch_id = 0
        for i in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1
        backward_micro_batch_id = 0
        for i in range(micro_batch_in_1f1b):
            for job_type in self.jobs_in_stable_phase:
                job = core.Job(job_type)
                micro_batch_id = forward_micro_batch_id if job_type.startswith(FORWARD) else backward_micro_batch_id
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
            forward_micro_batch_id += 1
            backward_micro_batch_id += 1
        for i in range(micro_batch_in_warmup):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1
        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _multistreaming_for_overlapping(self, programs, job_type):
        if False:
            while True:
                i = 10
        num_programs = len(programs)
        higher_stream_priority = -1
        for (program_id, program) in enumerate(programs):
            last_op = program.global_block().ops[-1]
            if self.is_comm_op_valid_to_overlap(last_op):
                last_op.dist_attr.execution_stream = AutoParallelStreamType.MP_STREAM.value
                last_op.dist_attr.stream_priority = higher_stream_priority
                prior_op_input_arg_names = last_op.input_arg_names
                prior_op_output_arg_names = last_op.output_arg_names
                for i in range(program_id + 1, num_programs):
                    posterior_ops = programs[i].global_block().ops
                    num_posterior_ops = len(posterior_ops)
                    for op_id in range(num_posterior_ops):
                        posterior_op = posterior_ops[op_id]
                        posterior_op_input_arg_names = posterior_op.input_arg_names
                        posterior_op_output_arg_names = posterior_op.output_arg_names
                        if set(prior_op_input_arg_names) & set(posterior_op_output_arg_names) or set(prior_op_output_arg_names) & set(posterior_op_input_arg_names) or set(prior_op_output_arg_names) & set(posterior_op_output_arg_names):
                            _add_event_dependency(last_op, posterior_op)

    def _op_cost(self, op):
        if False:
            return 10
        handwritten_cost_map = {'c_allreduce_sum': 0, 'elementwise_add': 40, 'split': 76, 'transpose2': 40, 'fused_softmax_mask_upper_triangle': 94, 'layer_norm': 55, 'gelu': 180, 'dropout': 160, 'c_identity': 0, 'recv_v2': 0}
        op_type = op.type
        if op_type in handwritten_cost_map.keys():
            return handwritten_cost_map[op_type]
        if op_type == 'matmul_v2':
            var_name = op.output_arg_names[0]
            shape = op.block._var_recursive(var_name).shape
            if shape == (1, 1024, 6144):
                return 399
            elif shape == (1, 16, 1024, 1024):
                return 112
            elif shape == (1, 16, 1024, 128):
                return 95
            elif shape == (1, 1024, 4096):
                return 244
        if op_type == 'scale':
            var_name = op.output_arg_names[0]
            shape = op.block._var_recursive(var_name).shape
            if shape == (1, 16, 1024, 128):
                return 20
            if shape == (1, 16, 1024, 1024):
                return 90
        try:
            time = calc_time_by_cost_model(op)
            if op.type == 'c_allreduce_sum':
                time *= 8
            return time
        except Exception as e:
            logger.info(f'The cost of {op} is unknown since {repr(e)}.')
            return 0.0

    def _partial_programs(self, program):
        if False:
            print('Hello World!')
        enable_send_recv_overlap = self.get_attr('enable_send_recv_overlap')
        types = [FORWARD, BACKWARD, OPT]
        sub_programs = _program_for_fthenb_and_1f1b(program, enable_send_recv_overlap)
        enable_backward_forward_overlap = self.get_attr('enable_backward_forward_overlap')
        if enable_backward_forward_overlap:
            logger.info('Backward forward overlap enabled in 1F1B.')
            (forward_program, backward_program) = (sub_programs[1], sub_programs[2])
            (splitted_backward_job_types, splitted_backward_programs, splitted_forward_job_types, splitted_forward_programs) = self._backward_forward_overlap(backward_program, forward_program)
            types += splitted_forward_job_types + splitted_backward_job_types
            sub_programs += splitted_forward_programs + splitted_backward_programs
        for i in range(len(types)):
            logger.debug(f'type = {types[i]}, sub_programs = {sub_programs[i]}\n')
        logger.debug(f'jobs_in_stable_phase = {self.jobs_in_stable_phase}')
        return (types, sub_programs)

    def _split_program_for_overlapping(self, job_type, program, split_points):
        if False:
            print('Hello World!')
        assert job_type in [FORWARD, BACKWARD], f'job_type should be one of {[FORWARD, BACKWARD]}'
        (splitted_programs, __, __) = split_program(program, split_points)
        splitted_job_types = []
        num_splitted_programs = len(splitted_programs)
        for idx in range(num_splitted_programs):
            splitted_job_types.append(f'{job_type}(chunk{idx})')
        return (splitted_job_types, splitted_programs)

    def is_comm_op_valid_to_overlap(self, op):
        if False:
            i = 10
            return i + 15
        return op.type == 'c_allreduce_sum' and op.dist_attr.execution_stream == AutoParallelStreamType.CALC_STREAM.value

@register_pass('pipeline_scheduler_Eager1F1B')
class PipelineEager1F1BPass(PipelinePassBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def _create_job_list(self):
        if False:
            return 10
        num_micro_batches = self.get_attr('num_micro_batches')
        pp_stage = self.get_attr('pp_stage')
        pp_degree = self.get_attr('pp_degree')
        job_list = []
        assert 2 * (pp_degree - pp_stage) - 1 <= num_micro_batches, 'Num of micro batches should larger than 2 * (pp_degree - pp_stage) - 1.'
        micro_batch_in_warmup = 2 * (pp_degree - pp_stage) - 1
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup
        forward_micro_batch_id = 0
        for _ in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1
        backward_micro_batch_id = 0
        for _ in range(micro_batch_in_1f1b):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1
        for _ in range(micro_batch_in_warmup):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1
        opt_job = core.Job(OPT)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        if False:
            print('Hello World!')
        enable_send_recv_overlap = self.get_attr('enable_send_recv_overlap')
        types = [FORWARD, BACKWARD, OPT]
        sub_program_list = _program_for_fthenb_and_1f1b(program, enable_send_recv_overlap)
        return (types, sub_program_list)

def apply_pass(main_program, startup_program, pass_name, pass_attr={}):
    if False:
        for i in range(10):
            print('nop')
    assert pass_name in ['FThenB', '1F1B', 'Eager1F1B'], f'pipeline scheduler only support FThenB, 1F1B and Eager1F1B, but recieve {pass_name}'
    if pass_name == '1F1B':
        pass_attr['enable_backward_forward_overlap'] = int(os.environ.get('FLAGS_1f1b_backward_forward_overlap', 0))
    pipeline_pass = new_pass('pipeline_scheduler_' + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply([main_program], [startup_program], pass_context)
    plan = pass_context.get_attr('plan')
    return plan