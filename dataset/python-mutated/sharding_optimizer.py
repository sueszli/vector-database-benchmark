import os
import paddle
from paddle.base import core
from paddle.incubate.optimizer import PipelineOptimizer
from paddle.static import create_global_var, default_startup_program, device_guard
from paddle.utils import unique_name
from ..utils.log_util import logger
from .common import OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, OpRole, is_backward_op, is_optimizer_op, is_update_op
from .meta_optimizer_base import MetaOptimizerBase
from .sharding import utils
from .sharding.fp16_helper import FP16Utils
from .sharding.gradient_clip_helper import GradientClipHelper
from .sharding.offload_helper import OffloadHelper
from .sharding.prune import ProgramDeps
from .sharding.shard import ProgramSegment, Shard
from .sharding.utils import get_first_optimize_op_idx, get_grad_device, get_var_size, insert_allreduce_ops, insert_broadcast_ops, insert_cast_ops, insert_fill_constant_ops, insert_reduce_ops, insert_scale_loss_grad_ops, insert_sync_calc_op, insert_sync_comm_ops
from .sharding.weight_decay_helper import WeightDecayHelper
__all__ = []

class ShardingOptimizer(MetaOptimizerBase):
    """Sharding Optimizer."""

    def __init__(self, optimizer):
        if False:
            return 10
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = ['RecomputeOptimizer', 'AMPOptimizer', 'LarsOptimizer', 'LambOptimizer', 'ASPOptimizer']
        self.meta_optimizers_black_list = []
        self._main_program = None
        self._startup_program = None
        self._segments = []
        self._params = set()
        self._broadcast_vars = set()
        self._reduced_grads_to_param = {}
        self._shard = Shard()
        self._verbose = False
        self.mp_degree = 1

    def _can_apply(self):
        if False:
            while True:
                i = 10
        if not self.role_maker._is_collective:
            return False
        if self.role_maker._worker_num() <= 1:
            return False
        return self.user_defined_strategy.sharding

    def _disable_strategy(self, dist_strategy):
        if False:
            print('Hello World!')
        dist_strategy.sharding = False
        dist_strategy.sharding_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        if False:
            while True:
                i = 10
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {'segment_broadcast_MB': 32}

    def _get_sharding_segment_strategy(self):
        if False:
            print('Hello World!')
        'get\n        self._sharding_segment_strategy\n        1. if by_size:    self._broadcast_MB\n        2. if by_anchors: self._sharding_segment_anchors\n                          self._backward_remain_anchors\n                          self._forward_remain_anchors\n        '
        strategy = self.user_defined_strategy
        sharding_configs = strategy.sharding_configs
        segment_strategy = str(sharding_configs['sharding_segment_strategy'])
        if segment_strategy == 'segment_broadcast_MB':
            self._broadcast_MB = sharding_configs['segment_broadcast_MB']
            assert self._broadcast_MB > 0, 'segment size should larger than zero !'
        elif segment_strategy == 'segment_anchors':
            self._sharding_segment_anchors = sharding_configs['segment_anchors']
            assert len(self._sharding_segment_anchors) > 0, 'you should set the sharding segment anchors !'
            self._backward_remain_anchors = self._sharding_segment_anchors[:]
            self._forward_remain_anchors = []
        else:
            raise NotImplementedError('the sharding segment strategy [{}] is not implemented'.format(str(segment_strategy)))
        self._sharding_segment_strategy = segment_strategy

    def _get_hybrid_degree(self):
        if False:
            i = 10
            return i + 15
        'get\n        self.hybrid_dp\n        self.sharding_degree\n        self.mp_degree\n        self.pp_degree\n        self.dp_degree\n        '
        strategy = self.user_defined_strategy
        sharding_configs = strategy.sharding_configs
        sharding_degree = int(sharding_configs['sharding_degree'])
        mp_degree = int(sharding_configs['mp_degree'])
        pp_degree = int(sharding_configs['pp_degree'])
        dp_degree = int(sharding_configs['dp_degree'])
        global_world_size = self.role_maker._worker_num()
        assert sharding_degree > 0, 'sharding degree must be larger than zero'
        if pp_degree > 1:
            assert strategy.pipeline is True
        if os.getenv('PADDLE_MANUAL_PIPELINE_STAGE', None):
            assert pp_degree == 2, 'For manually set pipeline, only pp_degree = 2 is supported.'
            assert global_world_size == mp_degree * sharding_degree * dp_degree, 'global work size [{}], mp_degree [{}], sharding_degree [{}], dp_degree [{}].'.format(global_world_size, mp_degree, sharding_degree, dp_degree)
        else:
            assert global_world_size == mp_degree * sharding_degree * pp_degree * dp_degree, 'global work size [{}], mp_degree [{}], sharding_degree [{}], pp_degree [{}], dp_degree [{}].'.format(global_world_size, mp_degree, sharding_degree, pp_degree, dp_degree)
        if sharding_configs['hybrid_dp']:
            logger.warning('[hybrid_dp] API setting is deprecated. Now when dp_degree >= 2, its will be in hybrid dp mode automatically')
            assert dp_degree >= 1
        self.hybrid_dp = True if dp_degree > 1 else False
        self.sharding_degree = sharding_degree
        self.mp_degree = mp_degree
        self.pp_degree = pp_degree
        self.dp_degree = dp_degree

    def _get_hybrid_dp_mode(self):
        if False:
            i = 10
            return i + 15
        "get\n        self.hybrid_dp_mode = 'pp_hybrid_dp' or 'sharding_hybrid_dp'\n        self.gradient_merge_mode = 'pp_gm' or 'sharding_gm'\n        self._gradient_merge_acc_step\n        self.pp_allreduce_in_optimize\n        self._optimizer_sharding\n        "
        strategy = self.user_defined_strategy
        sharding_configs = strategy.sharding_configs
        dp_mode = None
        if self.hybrid_dp:
            if self.pp_degree > 1:
                dp_mode = 'pp_hybrid_dp'
            else:
                assert self.sharding_degree > 1, 'by now we only support five kind of hybrid dp: sharding_hybrid_dp, mp_sharding_hybrid_dp, pp_hybrid_dp, mp_sharding_pp_hybrid_dp, sharding_pp_hybrid_dp.'
                dp_mode = 'sharding_hybrid_dp'
        gm_mode = None
        gm_acc_step = int(sharding_configs['gradient_merge_acc_step'])
        if self.pp_degree <= 1:
            gm_mode = 'sharding_gm'
            self._grad2merged_grad = {}
        else:
            gm_mode = 'pp_gm'
            gm_acc_step = strategy.pipeline_configs['accumulate_steps']
            gradient_scale_configs = strategy.gradient_scale_configs
            assert gradient_scale_configs['scale_strategy'] == 'avg', 'For pipeline mode, the gradient scale mode should be "avg", but got {}'.format(gradient_scale_configs['scale_strategy'])
            self.scale_gradient = gradient_scale_configs['scale_gradient']
        if gm_acc_step > 1:
            logger.info(f'Gradient merge in [{gm_mode}], acc step = [{gm_acc_step}]')
        optimizer_sharding = False
        if self.sharding_degree == 1 and self.dp_degree > 1 and sharding_configs['_dp_as_optimizer_sharding'] and (self.pp_degree > 1):
            optimizer_sharding = True
        self.hybrid_dp_mode = dp_mode
        self.gradient_merge_mode = gm_mode
        self._gradient_merge_acc_step = gm_acc_step
        self._optimizer_sharding = optimizer_sharding
        self.pp_allreduce_in_optimize = sharding_configs['pp_allreduce_in_optimize']

    def _inner_opt_minimize(self, loss, startup_program, parameter_list, no_grad_set):
        if False:
            i = 10
            return i + 15
        pipeline_configs = self.user_defined_strategy.pipeline_configs
        if self.inner_opt is None:
            raise ValueError('self.inner_opt of ShardingOptimizer should not be None.')
        if self.pp_degree > 1:
            pp_optimizer = PipelineOptimizer(self.inner_opt, self._gradient_merge_acc_step)
            self._pp_optimizer = pp_optimizer
            global_rank = self.role_maker._worker_index()
            schedule_mode = pipeline_configs['schedule_mode']
            pipeline_opt = {'schedule_mode': schedule_mode, 'micro_batch_size': pipeline_configs['micro_batch_size'], 'local_rank': self.pp_rank, 'global_rank': global_rank, 'use_sharding': True, 'ring_id': 20, 'global_ring_id': 3, 'mp_degree': self.mp_degree, 'mp_rank': global_rank % self.mp_degree, 'scale_gradient': self.scale_gradient}
            main_program = loss.block.program
            main_program._pipeline_opt = pipeline_opt
            (optimize_ops, params_grads, program_list, self.pipeline_pair, self.pp_ring_map) = pp_optimizer.minimize(loss, startup_program, parameter_list, no_grad_set)
            assert self.pp_degree == len(program_list)
        else:
            (optimize_ops, params_grads) = self.inner_opt.minimize(loss, startup_program, parameter_list, no_grad_set)
        if startup_program is None:
            startup_program = default_startup_program()
        if self.pp_degree > 1:
            startup_program = startup_program._pipeline_opt['startup_program']
            print('pp_rank:', self.pp_rank)
            if os.getenv('PADDLE_MANUAL_PIPELINE_STAGE', None):
                main_program = program_list[int(os.getenv('PADDLE_MANUAL_PIPELINE_STAGE'))]
            else:
                main_program = program_list[self.pp_rank]
            with open('main_%d' % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))
            main_block = main_program.global_block()
            new_params_grads = []
            for (param, grad) in params_grads:
                if main_block.has_var(param.name):
                    new_params_grads.append((param, grad))
            params_grads = new_params_grads
        else:
            main_block = loss.block
        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program
        if self.pp_degree > 1:
            pp_optimizer._rename_gradient_var_name(main_block)
            with open('main_%d' % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))
        return (optimize_ops, params_grads)

    def _apply_sharding_pass(self, params_grads):
        if False:
            i = 10
            return i + 15
        if self.sharding_degree == 1:
            return
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        self._build_shard(params_grads, self.sharding_rank, self.sharding_degree)
        self._split_program(main_block)
        self._add_broadcast_allreduce(main_block)
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()
        self._prune_main_program(main_block, self._shard, [self.mp_ring_id, self.sharding_ring_id, self.pp_ring_id])
        self._prune_startup_program(startup_block, self._shard)

    def _apply_opt_sharding_pass(self, params_grads):
        if False:
            return 10
        'outer dp as optimizer sharding'
        if self._optimizer_sharding is False:
            return
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        self._build_shard(params_grads, self.dp_rank, self.dp_degree)
        for (param, grad) in params_grads:
            self._reduced_grads_to_param[grad.name] = param.name
        self._prune_main_program(main_block, self._shard, [self.mp_ring_id, self.pp_ring_id, self.dp_ring_id])
        self._prune_startup_program(startup_block, self._shard)

    def _insert_allreduce_for_pp(self, params_grads):
        if False:
            while True:
                i = 10
        if self.pp_degree == 1:
            return
        strategy = self.user_defined_strategy
        sharding_configs = strategy.sharding_configs
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        if self.sharding_degree > 1:
            for (idx, op) in reversed(list(enumerate(main_block.ops))):
                if is_update_op(op):
                    op_role_var = op.attr('op_role_var')
                    param_name = op_role_var[0]
                    if not self._shard.has_param(param_name):
                        main_block._remove_op(idx)
            for (idx, op) in reversed(list(enumerate(main_block.ops))):
                if op.type != 'cast':
                    continue
                in_name = op.input_arg_names[0]
                if in_name not in self._params:
                    continue
                if in_name not in main_block.vars:
                    main_block._remove_op(idx)
        if self._optimizer_sharding:
            strategy.fp16_allreduce = False
        shard = self._shard if self._optimizer_sharding else None
        accumulated_grad_names = self._pp_optimizer._accumulate_gradients(main_block, strategy=strategy, shard=shard)
        len_of_ops = len(main_block.ops)
        if self.scale_gradient:
            self._avg_grad_merge_after_sum(main_block, accumulated_grad_names)
        first_optimize_op_index = get_first_optimize_op_idx(main_block)
        if self.pp_allreduce_in_optimize:
            logger.info(f'Pipeline Persistable grad is {accumulated_grad_names}')
            accumulated_grad_names = insert_reduce_ops(main_block, first_optimize_op_index, self.sharding_ring_id, accumulated_grad_names, self._shard, core.op_proto_and_checker_maker.OpRole.Optimize, use_calc_stream=True, rank=self.sharding_rank)
            logger.info(f'PP-Sharding grad is {accumulated_grad_names}')
            first_optimize_op_index += len(main_block.ops) - len_of_ops
            len_of_ops = len(main_block.ops)
        if self._optimizer_sharding:
            accumulated_grad_names = utils.insert_reduce_ops(main_block, first_optimize_op_index, self.dp_ring_id, accumulated_grad_names, self._shard, OpRole.Optimize, use_calc_stream=True, rank=self.dp_rank, strategy=strategy)
            logger.info(f'Optimizer grad in this rank {accumulated_grad_names}')
            first_optimize_op_index += len(main_block.ops) - len_of_ops
            len_of_ops = len(main_block.ops)
            optimize_cast = sharding_configs['optimize_cast']
            optimizer_param = utils.insert_broadcast_param_ops(main_block, len_of_ops, self.dp_ring_id, [x[0].name for x in params_grads], self._shard, OpRole.Optimize, use_calc_stream=True, rank=self.dp_rank, strategy=None if optimize_cast else strategy)
            logger.info(f'Optimizer param in this rank {optimizer_param}')
            if not strategy.fuse_grad_merge and (not optimize_cast):
                assert len(accumulated_grad_names) == len(optimizer_param)
        elif self.hybrid_dp and self.hybrid_dp_mode == 'pp_hybrid_dp':
            insert_allreduce_ops(main_block, first_optimize_op_index, self.dp_ring_id, accumulated_grad_names, core.op_proto_and_checker_maker.OpRole.Optimize, use_calc_stream=True, user_defined_strategy=strategy)
            first_optimize_op_index += len(main_block.ops) - len_of_ops
            len_of_ops = len(main_block.ops)

    def _avg_grad_merge_after_sum(self, main_block, accumulated_grad_names):
        if False:
            i = 10
            return i + 15
        if self.user_defined_strategy.amp and self.user_defined_strategy.amp_configs['use_dynamic_loss_scaling']:
            for (idx, op) in enumerate(main_block.ops):
                if op.type == 'check_finite_and_unscale':
                    loss_scale_name = op.input('Scale')[0]
                    loss_scaling_var = main_block.var(loss_scale_name)
                    loss_scale_tmp_var_name = loss_scale_name + '@TMP'
                    loss_scale_tmp_var = main_block.create_var(name=loss_scale_tmp_var_name, shape=loss_scaling_var.shape, dtype=loss_scaling_var.dtype)
                    main_block._insert_op_without_sync(idx, type='scale', inputs={'X': loss_scaling_var}, outputs={'Out': loss_scale_tmp_var}, attrs={'scale': self._gradient_merge_acc_step, 'bias': 0.0, 'bias_after_scale': False, OP_ROLE_KEY: OpRole.Optimize})
                    op._rename_input(loss_scale_name, loss_scale_tmp_var_name)
                    break
        else:
            tmp_first_opt_idx = None
            for (idx, op) in enumerate(main_block.ops):
                if is_optimizer_op(op) and op.type != 'c_sync_comm_stream':
                    tmp_first_opt_idx = idx
                    break
            assert tmp_first_opt_idx is not None, 'Occurs some errors, no optimize ops'
            for grad in accumulated_grad_names:
                main_block._insert_op_without_sync(tmp_first_opt_idx, type='scale', inputs={'X': grad}, outputs={'Out': grad}, attrs={'scale': 1.0 / self._gradient_merge_acc_step, 'bias': 0.0, 'bias_after_scale': False, OP_ROLE_KEY: OpRole.Optimize})

    def _adapt_amp_clip_without_sharding(self):
        if False:
            return 10
        if self.sharding_degree > 1:
            return
        if self._optimizer_sharding:
            return
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        rings = [self.mp_ring_id, self.pp_ring_id]
        FP16Utils.sync_amp_check_nan_inf(main_block, rings)
        gradientclip_helper = GradientClipHelper(None)
        gradientclip_helper.sync_global_norm(main_block, [self.mp_ring_id, self.pp_ring_id], self.mp_rank)

    def _insert_loss_grad_scale_op(self):
        if False:
            i = 10
            return i + 15
        main_block = self._main_program.global_block()
        global_dp_degree = self.sharding_degree * self.dp_degree
        assert int(global_dp_degree) == global_dp_degree
        if global_dp_degree > 1:
            insert_scale_loss_grad_ops(main_block, scale=global_dp_degree)
        main_block._sync_with_cpp()

    def _apply_optimize_offload_pass(self, params_grads):
        if False:
            while True:
                i = 10
        strategy = self.user_defined_strategy
        sharding_configs = strategy.sharding_configs
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        mp_ring_id = self.mp_ring_id if self.mp_degree > 1 else None
        dp_ring_id = self.dp_ring_id if self.dp_degree > 1 else None
        offload_helper = OffloadHelper(mp_ring_id=mp_ring_id, dp_ring_id=dp_ring_id)
        if sharding_configs['optimize_offload']:
            logger.info('Sharding with optimize offload !')
            offload_helper.offload(main_block, startup_block)
            offload_helper.offload_fp32param(main_block, startup_block)
        elif sharding_configs['optimize_cast']:
            logger.info('Sharding with optimize cast !')
            if self._optimizer_sharding:
                offload_helper.opt_sharding_cast_fp32param(main_block, startup_block, [x[0].name for x in params_grads])
                utils.fuse_opt_broadcast_param_ops(main_block, dp_ring_id, self._shard, strategy=strategy)
            else:
                offload_helper.cast_fp32param_in_optimize(main_block, startup_block)

    def _dump_program_for_debug(self):
        if False:
            for i in range(10):
                print('nop')
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()
        with open('start_sharding_%d' % self.role_maker._worker_index(), 'w') as f:
            f.writelines(str(startup_block.program))
        with open('main_sharding_%d' % self.role_maker._worker_index(), 'w') as f:
            f.writelines(str(main_block.program))

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            print('Hello World!')
        self._nrings_sharding = 1
        self._nrings_dp = 1
        self._get_sharding_segment_strategy()
        self._get_hybrid_degree()
        self._get_hybrid_dp_mode()
        self._build_groups()
        (optimize_ops, params_grads) = self._inner_opt_minimize(loss, startup_program, parameter_list, no_grad_set)
        self._init_comm()
        self._apply_sharding_pass(params_grads)
        self._apply_opt_sharding_pass(params_grads)
        self._insert_allreduce_for_pp(params_grads)
        self._adapt_amp_clip_without_sharding()
        self._insert_loss_grad_scale_op()
        self._apply_optimize_offload_pass(params_grads)
        self._sharding_gradient_merge()
        self._initialization_broadcast()
        self._recreate_not_persist_param_as_var()
        self._dump_program_for_debug()
        use_new_comm = paddle.get_flags('FLAGS_dynamic_static_unified_comm')['FLAGS_dynamic_static_unified_comm']
        if not use_new_comm:
            self._wait()
        return (optimize_ops, params_grads)

    def _init_pair_comm(self, pair, ring_id):
        if False:
            i = 10
            return i + 15
        pp_group_endpoints = [self.pp_group_endpoints[pair[0]], self.pp_group_endpoints[pair[1]]]
        pp_rank = 0 if self.pp_rank == pair[0] else 1
        if os.getenv('PADDLE_MANUAL_PIPELINE_STAGE', None) is None:
            self._collective_helper._init_communicator(self._startup_program, self.current_endpoint, pp_group_endpoints, pp_rank, ring_id, False, sync=False)

    def _init_pipeline_comm(self, startup_block):
        if False:
            while True:
                i = 10
        if os.getenv('PADDLE_MANUAL_PIPELINE_STAGE', None) is None:
            self._collective_helper._init_communicator(self._startup_program, self.current_endpoint, self.pp_group_endpoints, self.pp_rank, self.pp_ring_id, False, sync=False)
        for pair in self.pipeline_pair:
            pair_key = pair[0] * 1000 + pair[1]
            ring_id = self.pp_ring_map[pair_key]
            logger.info(f'pp pair:{pair}, ring_id: {ring_id}')
            if self.pp_rank in pair:
                self._init_pair_comm(pair, ring_id)

    def _init_comm(self):
        if False:
            for i in range(10):
                print('nop')
        startup_block = self._startup_program.global_block()
        if self.mp_degree > 1:
            self._collective_helper._init_communicator(self._startup_program, self.current_endpoint, self.mp_group_endpoints, self.mp_rank, self.mp_ring_id, False, sync=False)
        if self.sharding_degree > 1:
            self._collective_helper._init_communicator(self._startup_program, self.current_endpoint, self.sharding_group_endpoints, self.sharding_rank, self.sharding_ring_id, False, sync=False)
        if self.pp_degree > 1:
            self._init_pipeline_comm(startup_block)
        if self.dp_degree > 1:
            self._collective_helper._init_communicator(self._startup_program, self.current_endpoint, self.dp_group_endpoints, self.dp_rank, self.dp_ring_id, False, sync=False)
        startup_block._sync_with_cpp()

    def _build_shard(self, params_grads, shard_rank, shard_size):
        if False:
            return 10
        self._params = {x[0].name for x in params_grads}
        self._shard.setup(params_grads, shard_rank, shard_size)
        self._broadcast_vars = self._shard.find_broadcast_params(self._main_program.global_block())

    def _wait(self):
        if False:
            print('Hello World!')
        endpoints = self.global_endpoints[:]
        current_endpoint = endpoints[self.global_rank]
        if self.global_rank == 0:
            self._collective_helper._wait(current_endpoint, endpoints)

    def collect_segment(self, segment, op_idx, block):
        if False:
            return 10
        segment._start_idx = op_idx + 1
        self._segments.insert(0, segment)
        new_segment = ProgramSegment(block)
        new_segment._end_idx = op_idx + 1
        return new_segment

    def _split_program(self, block):
        if False:
            for i in range(10):
                print('nop')
        for (op_idx, op) in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                last_backward_op_idx = op_idx + 1
                break
        var2broadcast_time = {}
        segment = ProgramSegment(block)
        segment._end_idx = last_backward_op_idx
        for op_idx in reversed(range(last_backward_op_idx)):
            op = block.ops[op_idx]
            assert int(op.attr('op_role')) != int(OpRole.Optimize)
            if self._sharding_segment_strategy == 'segment_broadcast_MB':
                if segment._param_mem >= self._broadcast_MB:
                    segment = self.collect_segment(segment, op_idx, block)
            elif self._sharding_segment_strategy == 'segment_anchors':
                if int(op.attr('op_role')) == int(OpRole.Backward):
                    for input_name in op.desc.input_arg_names():
                        if self.user_defined_strategy.amp:
                            if '.cast_fp16@GRAD' not in input_name:
                                continue
                            else:
                                input_name = input_name[:input_name.find('.cast_fp16@GRAD')]
                        if input_name in self._backward_remain_anchors:
                            segment = self.collect_segment(segment, op_idx, block)
                            assert input_name not in self._forward_remain_anchors, f'segment anchor [{input_name}] met twice !'
                            self._backward_remain_anchors.remove(input_name)
                            self._forward_remain_anchors.append(input_name)
                elif int(op.attr('op_role')) == int(OpRole.Forward):
                    for output_name in op.desc.output_arg_names():
                        if output_name in self._forward_remain_anchors:
                            segment = self.collect_segment(segment, op_idx, block)
                            self._forward_remain_anchors.remove(output_name)
            for input_name in op.desc.input_arg_names():
                if input_name not in self._broadcast_vars:
                    continue
                if input_name in segment._param2broadcast:
                    broadcast_name = segment._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if self._shard.has_param(input_name):
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name + '@BroadCast')
                    segment._fill_constant_vars.append(broadcast_var_name)
                broadcast_var_base_name = input_name
                if 'subprog' in broadcast_var_base_name:
                    broadcast_var_base_name = broadcast_var_base_name[:broadcast_var_base_name.find('.subprog')]
                var2broadcast_time[broadcast_var_base_name] = var2broadcast_time.get(broadcast_var_base_name, 0) + 1
                segment._param2broadcast[input_name] = broadcast_var_name
                segment._broadcast_vars.append((broadcast_var_name, self._shard.device(input_name)))
                segment._param_mem += get_var_size(self._main_program.global_block().var(input_name))
            if self.pp_degree > 1 and self.pp_allreduce_in_optimize:
                pass
            elif is_backward_op(op) and OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) != 0:
                    assert len(op_role_var) % 2 == 0
                    for i in range(0, len(op_role_var), 2):
                        (param, reduced_grad) = (op_role_var[i], op_role_var[i + 1])
                        segment._allreduce_vars.append(reduced_grad)
                        assert reduced_grad not in self._reduced_grads_to_param
                        self._reduced_grads_to_param[reduced_grad] = param
            if FP16Utils.is_fp16_cast_op(block, op, self._params):
                fp32_param = op.desc.input_arg_names()[0]
                fp16_param = op.desc.output_arg_names()[0]
                if self._shard.has_param(fp32_param):
                    segment._cast_ops[fp16_param] = fp32_param
        if segment._param_mem > 0:
            segment._start_idx = 0
            self._segments.insert(0, segment)
        if self._sharding_segment_strategy == 'segment_anchors':
            assert len(self._forward_remain_anchors) == 0, f'remain anchors {self._forward_remain_anchors}'
            assert len(self._backward_remain_anchors) == 0, f'remain anchors {self._backward_remain_anchors}'
        if self._verbose:
            for varname in sorted(var2broadcast_time, key=var2broadcast_time.get, reverse=True):
                logger.info('Sharding broadcast: [{}] times [{}]'.format(var2broadcast_time[varname], varname))
            for idx_ in range(len(self._segments)):
                logger.info(f'segment [{idx_}] :')
                logger.info('start op: [{}]  [{}]'.format(block.ops[self._segments[idx_]._start_idx].desc.type(), block.ops[self._segments[idx_]._start_idx].desc.input_arg_names()))
                logger.info('end   op: [{}]  [{}]'.format(block.ops[self._segments[idx_]._end_idx].desc.type(), block.ops[self._segments[idx_]._end_idx].desc.input_arg_names()))

    def _prune_main_program(self, block, shard, rings):
        if False:
            while True:
                i = 10
        '\n        calculate deps from allredce op to optimize op,\n        remove ops and vars not needed in this worker\n\n        1. prune regularization (weight decay)\n        2. prune cast_fp32_to_fp16; update amp_infine_checking\n        3. prune gradient_clip related; update global_norm_sum\n        4. prune optimizer op + param + gradient\n\n        '
        weightdecay_helper = WeightDecayHelper()
        weightdecay_helper.prune_weight_decay(block, shard)
        FP16Utils.prune_fp16(block, shard, self._reduced_grads_to_param, rings)
        gradientclip_helper = GradientClipHelper(None)
        gradientclip_helper.prune_gradient_clip(block, shard, rings)
        reduced_grads = []
        for (idx, op) in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == 'c_allreduce_sum' and op.attr('use_model_parallel') is False:
                assert len(output_names) == 1
                output_name = output_names[0]
                reduced_grads.append(output_name)
        pruned_opti_vars = []
        for var_name in list(block.vars.keys()):
            if shard.is_opti_var(var_name) and (not shard.has_opt_var(var_name)):
                pruned_opti_vars.append(var_name)
        program_deps = ProgramDeps(block, reduced_grads, pruned_opti_vars)
        for var_name in program_deps._end_vars:
            program_deps._should_removed_var.add(var_name)
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if op.type in ['c_allreduce_sum', 'c_sync_comm_stream', 'c_calc_comm_stream', 'c_gen_nccl_id', 'c_gen_bkcl_id', 'c_gen_xccl_id', 'c_comm_init', 'send_v2', 'recv_v2']:
                pass
            elif op.type == 'conditional_block':
                assert op.desc.has_attr('sub_block')
                subblock_idx = op.desc.attr('sub_block').id
                subblock_deps = program_deps.get_sub_block_deps(subblock_idx)
                if subblock_deps is None or not self._is_amp_subblock(op):
                    continue
                reversed_output_vars = []
                for output_name in op.desc.output('Out'):
                    if output_name in program_deps._should_removed_var:
                        subblock_deps._should_removed_var.add(output_name)
                        program_deps.crop_output_var_from_op(idx, output_name)
                    else:
                        reversed_output_vars.append(output_name)
                for (sub_op_idx, _) in reversed(list(enumerate(subblock_deps._block.ops))):
                    if subblock_deps.should_remove_op(sub_op_idx):
                        subblock_deps.remove_op(sub_op_idx)
                reversed_input_vars = []
                for input_name in op.desc.input('Input'):
                    if input_name not in subblock_deps._should_removed_var:
                        reversed_input_vars.append(input_name)
                    else:
                        program_deps.crop_input_var_from_op(idx, input_name)
                op.desc.set_input('Input', reversed_input_vars)
                op.desc.set_output('Out', reversed_output_vars)
            elif program_deps.should_remove_op(idx):
                reserved_vars = self._params if self._optimizer_sharding else None
                program_deps.remove_op(idx, reserved_vars)
        block._sync_with_cpp()
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if op.type == 'concat' and is_optimizer_op(op):
                reserved_x = []
                for var_name in op.desc.input('X'):
                    if block.has_var(var_name):
                        reserved_x.append(var_name)
                op.desc.set_input('X', reserved_x)
        block._sync_with_cpp()

    def _add_broadcast_allreduce(self, block):
        if False:
            return 10
        '\n        add broadcast allreduce op\n        if enable gradient_merge, insert related ops\n\n        if combined with pipeline(grad accumulate),\n        the grad allreduce should be done in optimize role\n        '
        if len(self._segments) < 1:
            return
        if self.pp_degree > 1 and self.pp_allreduce_in_optimize:
            for idx in range(len(self._segments)):
                assert len(self._segments[idx]._allreduce_vars) == 0
        new_end_idx = self._segments[-1]._end_idx
        for idx in range(self._segments[-1]._end_idx - 1, self._segments[-1]._start_idx - 1, -1):
            op = block.ops[idx]
            if op.type == 'fill_constant' or op.type == 'sum':
                if 'MERGED' in op.output_arg_names[0]:
                    new_end_idx = idx + 1
            elif op.type == 'cast':
                if '@TMP' in op.output_arg_names[0]:
                    new_end_idx = idx + 1
        self._segments[-1]._end_idx = new_end_idx
        if self._segments[-1]._allreduce_vars:
            shard_allredue_vars = self._shard.filter_grads(self._segments[-1]._allreduce_vars)
            if self.gradient_merge_mode != 'sharding_gm' or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == 'sharding_hybrid_dp' and (len(shard_allredue_vars) >= 1):
                    insert_sync_comm_ops(block, self._segments[-1]._end_idx, self.dp_ring_id, shard_allredue_vars)
                    insert_allreduce_ops(block, self._segments[-1]._end_idx, self.dp_ring_id, shard_allredue_vars, user_defined_strategy=self.user_defined_strategy)
            elif self.gradient_merge_mode == 'sharding_gm' and self._gradient_merge_acc_step > 1:
                self.create_persistable_gradients_and_insert_merge_ops(block, self._startup_program.global_block(), self._segments[-1]._end_idx, shard_allredue_vars, self._shard)
            insert_sync_comm_ops(block, self._segments[-1]._end_idx, self.sharding_ring_id, self._segments[-1]._allreduce_vars)
            insert_reduce_ops(block, self._segments[-1]._end_idx, self.sharding_ring_id, self._segments[-1]._allreduce_vars, self._shard, op_role=OpRole.Backward, use_calc_stream=False)
        for (idx, segment) in reversed(list(enumerate(self._segments))):
            allreduce_vars = self._segments[idx - 1]._allreduce_vars if idx > 0 else []
            broadcast_vars = self._segments[idx + 1]._broadcast_vars if idx < len(self._segments) - 1 else []
            fill_constant_vars = self._segments[idx + 2]._fill_constant_vars if idx < len(self._segments) - 2 else []
            cast_ops = self._segments[idx + 2]._cast_ops if idx < len(self._segments) - 2 else {}
            for op_idx in reversed(range(segment._start_idx, segment._end_idx)):
                op = block.ops[op_idx]
                for input_name in op.desc.input_arg_names():
                    if input_name in segment._param2broadcast and input_name != segment._param2broadcast[input_name]:
                        op._rename_input(input_name, segment._param2broadcast[input_name])
            for (param_name, broadcast_name) in segment._param2broadcast.items():
                if param_name != broadcast_name:
                    block.create_var(name=broadcast_name, shape=self._main_program.global_block().var(param_name).shape, dtype=self._main_program.global_block().var(param_name).dtype, persistable=False)
            block._sync_with_cpp()
            segment._end_idx += FP16Utils.remove_cast_op(block, self._params, segment, 0)
            shard_allredue_vars = self._shard.filter_grads(allreduce_vars)
            if self.gradient_merge_mode != 'sharding_gm' or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == 'sharding_hybrid_dp' and (len(shard_allredue_vars) >= 1):
                    insert_sync_comm_ops(block, segment._end_idx, self.dp_ring_id, shard_allredue_vars)
                    broad_cast_vars = [x[0] for x in broadcast_vars]
                    if len(broad_cast_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx, self.sharding_ring_id, broad_cast_vars)
                else:
                    comm_dep_vars = allreduce_vars + [x[0] for x in broadcast_vars]
                    if len(comm_dep_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx, self.sharding_ring_id, comm_dep_vars)
            elif self.gradient_merge_mode == 'sharding_gm' and self._gradient_merge_acc_step > 1:
                broad_cast_vars = [x[0] for x in broadcast_vars]
                if len(broad_cast_vars) > 0:
                    insert_sync_comm_ops(block, segment._end_idx, self.sharding_ring_id, broad_cast_vars)
            calc_dep_vars = fill_constant_vars + [k for (k, v) in cast_ops.items()] + self._segments[idx]._allreduce_vars
            if len(calc_dep_vars) > 0:
                insert_sync_calc_op(block, segment._end_idx, [calc_dep_vars[-1]])
            insert_fill_constant_ops(block, segment._end_idx, fill_constant_vars)
            insert_cast_ops(block, segment._end_idx, cast_ops)
            if self.gradient_merge_mode == 'sharding_gm' and self._gradient_merge_acc_step > 1:
                self.create_persistable_gradients_and_insert_merge_ops(block, self._startup_program.global_block(), segment._start_idx, shard_allredue_vars, self._shard)
            insert_broadcast_ops(block, segment._start_idx, self.sharding_ring_id, broadcast_vars)
            if self.gradient_merge_mode != 'sharding_gm' or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == 'sharding_hybrid_dp' and (len(shard_allredue_vars) >= 1):
                    insert_allreduce_ops(block, segment._start_idx, self.dp_ring_id, shard_allredue_vars, user_defined_strategy=self.user_defined_strategy)
                    insert_sync_comm_ops(block, segment._start_idx, self.sharding_ring_id, allreduce_vars)
            elif self.gradient_merge_mode == 'sharding_gm' and self._gradient_merge_acc_step > 1:
                insert_sync_comm_ops(block, segment._start_idx, self.sharding_ring_id, allreduce_vars)
            if len(allreduce_vars) > 0:
                insert_reduce_ops(block, segment._start_idx, self.sharding_ring_id, allreduce_vars, self._shard, op_role=OpRole.Backward, use_calc_stream=False)
            block._sync_with_cpp()
        if self._segments[0]._broadcast_vars:
            broadcast_vars = [x[0] for x in self._segments[0]._broadcast_vars]
            insert_sync_comm_ops(block, self._segments[0]._start_idx, self.sharding_ring_id, broadcast_vars)
            insert_broadcast_ops(block, self._segments[0]._start_idx, self.sharding_ring_id, self._segments[0]._broadcast_vars)
        fill_constant_vars = []
        for x in self._segments[:2]:
            fill_constant_vars += x._fill_constant_vars
        cast_ops = {}
        for x in self._segments[:2]:
            for (k, v) in x._cast_ops.items():
                cast_ops[k] = v
        calc_deps_vars = fill_constant_vars + [k for (k, v) in cast_ops.items()]
        if fill_constant_vars or cast_ops:
            insert_sync_calc_op(block, self._segments[0]._start_idx, [calc_deps_vars[-1]])
        if fill_constant_vars:
            insert_fill_constant_ops(block, self._segments[0]._start_idx, fill_constant_vars)
        if cast_ops:
            insert_cast_ops(block, self._segments[0]._start_idx, cast_ops)
        return

    def _prune_startup_program(self, block, shard):
        if False:
            i = 10
            return i + 15
        for (idx, op) in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                if shard.has_var(output_name):
                    continue
                if self._optimizer_sharding and shard.is_param(output_name):
                    continue
                block._remove_op(idx, sync=False)
                break
        for var_name in list(block.vars.keys()):
            if shard.has_var(var_name):
                continue
            if self._optimizer_sharding and shard.is_param(var_name):
                continue
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()

    def _build_groups(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        pre-assign ring ids\n            mp: 0\n            sharding: 1\n            pure-dp: 2\n            global: 3\n            pp: 4\n            pp-pair: >= 20\n        if one parallelism is not enable: -1\n        and only support parallelism hierarchy: mp --> sharding --> pp --> dp\n        '
        self.global_word_size = self.role_maker._worker_num()
        self.global_rank = self.role_maker._worker_index()
        self.global_endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.global_endpoints[self.global_rank]
        self._collective_helper = CollectiveHelper(self.role_maker, nrings=self._nrings_sharding)
        assert self.global_word_size % self.mp_degree == 0, 'global_word_size: {} should be divisible to the mp_degree: {}'.format(self.global_word_size, self.mp_degree)
        assert self.global_word_size % self.sharding_degree == 0, 'global_word_size: {} should be divisible to the sharding_degree: {}'.format(self.global_word_size, self.sharding_degree)
        assert self.global_word_size % self.pp_degree == 0, 'global_word_size: {} should be divisible to the pp_degree: {}'.format(self.global_word_size, self.pp_degree)
        assert self.global_word_size % self.dp_degree == 0, 'global_word_size: {} should be divisible to the dp_degree: {}'.format(self.global_word_size, self.dp_degree)
        if self.mp_degree > 1:
            self.mp_ring_id = 0
            self.mp_rank = self.global_rank % self.mp_degree
            self.mp_group_id = self.global_rank // self.mp_degree
            self.mp_group_endpoints = [ep for (idx, ep) in enumerate(self.global_endpoints) if idx // self.mp_degree == self.mp_group_id]
            assert self.current_endpoint in self.mp_group_endpoints
            assert len(self.mp_group_endpoints) == self.mp_degree, 'num of mp worker in group is [{}], but mp group size is [{}]'.format(len(self.mp_group_endpoints), self.mp_degree)
        else:
            self.mp_degree = 1
            self.mp_ring_id = -1
            self.mp_rank = -1
            self.mp_group_id = -1
            self.mp_group_endpoints = []
        if self.sharding_degree > 1:
            self.sharding_ring_id = 1
            self.sharding_rank = self.global_rank // self.mp_degree % self.sharding_degree
            self.sharding_group_id = self.global_rank // (self.mp_degree * self.sharding_degree)
            if self.mp_degree > 1:
                self.sharding_group_endpoints = [ep for (idx, ep) in enumerate(self.global_endpoints) if idx // (self.mp_degree * self.sharding_degree) == self.sharding_group_id and idx % self.mp_degree == self.mp_rank]
            else:
                self.sharding_group_endpoints = [ep for (idx, ep) in enumerate(self.global_endpoints) if idx // (self.mp_degree * self.sharding_degree) == self.sharding_group_id]
            assert self.current_endpoint in self.sharding_group_endpoints
        else:
            self.sharding_degree = 1
            self.sharding_ring_id = -1
            self.sharding_rank = -1
            self.sharding_group_id = -1
            self.sharding_group_endpoints = []
        if self.pp_degree > 1:
            self.pp_pair_ring_id = 20
            self.pp_ring_id = 4
            self.pp_rank = self.global_rank // (self.sharding_degree * self.mp_degree) % self.pp_degree
            self.pp_group_id = self.global_rank // (self.mp_degree * self.sharding_degree * self.pp_degree)
            pp_first_stage_idx = self.global_rank % (self.sharding_degree * self.mp_degree) + self.pp_group_id * (self.mp_degree * self.sharding_degree * self.pp_degree)
            pp_stage_offset = self.sharding_degree * self.mp_degree
            self.pp_group_endpoints = []
            for i in range(self.pp_degree):
                self.pp_group_endpoints.append(self.global_endpoints[pp_first_stage_idx + pp_stage_offset * i])
            assert self.current_endpoint in self.pp_group_endpoints
        else:
            self.pp_ring_id = -1
            self.pp_degree = 1
            self.pp_pair_ring_id = -1
            self.pp_rank = -1
            self.pp_group_id = -1
            self.pp_group_endpoints = []
        local_pp_degree = self.pp_degree
        if os.getenv('PADDLE_MANUAL_PIPELINE_STAGE', None):
            assert self.pp_degree == 2, 'For manually set pipeline, only pp_degree = 2 is supported.'
            assert self.global_word_size == self.mp_degree * self.sharding_degree * self.dp_degree, 'global work size [{}], mp_degree [{}], sharding_degree [{}], dp_degree [{}].'.format(self.global_word_size, self.mp_degree, self.sharding_degree, self.dp_degree)
            local_pp_degree = 1
        else:
            assert self.global_word_size == self.mp_degree * self.sharding_degree * self.pp_degree * self.dp_degree, 'mp_degree: [{}], sharding_degree: [{}], pp_degree: [{}], dp_degree: [{}]; BUT global nrank: [{}]'.format(self.mp_degree, self.sharding_degree, self.pp_degree, self.dp_degree, self.global_word_size)
        if self.dp_degree > 1:
            self.dp_ring_id = 2
            self.dp_rank = self.global_rank // (self.sharding_degree * self.mp_degree * local_pp_degree)
            dp_first_rank_idx = self.global_rank % (self.sharding_degree * self.mp_degree * local_pp_degree)
            dp_offset = self.sharding_degree * self.mp_degree * local_pp_degree
            self.dp_group_endpoints = []
            for i in range(self.dp_degree):
                self.dp_group_endpoints.append(self.global_endpoints[dp_first_rank_idx + dp_offset * i])
            assert self.current_endpoint in self.dp_group_endpoints
            logger.info('Hybrid DP mode turn on !')
        else:
            self.dp_ring_id = -1
            self.dp_rank = -1
            self.dp_group_endpoints = []
        self.global_ring_id = 3
        logger.info(f'global word size: {self.global_word_size}')
        logger.info(f'global rank: {self.global_rank}')
        logger.info(f'global endpoints: {self.global_endpoints}')
        logger.info(f'global ring id: {self.global_ring_id}')
        logger.info('#####' * 6)
        logger.info(f'mp group size: {self.mp_degree}')
        logger.info(f'mp rank: {self.mp_rank}')
        logger.info(f'mp group id: {self.mp_group_id}')
        logger.info(f'mp group endpoints: {self.mp_group_endpoints}')
        logger.info(f'mp ring id: {self.mp_ring_id}')
        logger.info('#####' * 6)
        logger.info(f'sharding group size: {self.sharding_degree}')
        logger.info(f'sharding rank: {self.sharding_rank}')
        logger.info(f'sharding group id: {self.sharding_group_id}')
        logger.info(f'sharding group endpoints: {self.sharding_group_endpoints}')
        logger.info(f'sharding ring id: {self.sharding_ring_id}')
        logger.info('#####' * 6)
        logger.info(f'pp group size: {self.pp_degree}')
        logger.info(f'pp rank: {self.pp_rank}')
        logger.info(f'pp group id: {self.pp_group_id}')
        logger.info(f'pp group endpoints: {self.pp_group_endpoints}')
        logger.info(f'pp ring id: {self.pp_ring_id}')
        logger.info('#####' * 6)
        logger.info(f'pure dp group size: {self.dp_degree}')
        logger.info(f'pure dp rank: {self.dp_rank}')
        logger.info(f'pure dp group endpoints: {self.dp_group_endpoints}')
        logger.info(f'pure dp ring id: {self.dp_ring_id}')
        logger.info('#####' * 6)

    def _recreate_not_persist_param_as_var(self):
        if False:
            for i in range(10):
                print('nop')

        def recreate_not_persist_param_as_var(program):
            if False:
                for i in range(10):
                    print('nop')
            block = program.global_block()
            params = block.all_parameters()
            for param in params:
                if param.persistable:
                    continue
                name = param.name
                shape = param.shape
                dtype = param.dtype
                type = param.type
                lod_level = param.lod_level
                stop_gradient = param.stop_gradient
                trainable = param.trainable
                optimize_attr = param.optimize_attr
                regularizer = param.regularizer
                have_dist_attr = False
                is_distributed = False
                if hasattr(param, 'is_distributed'):
                    have_dist_attr = True
                    is_distributed = param.is_distributed
                block._remove_var(name, sync=False)
                var = block.create_var(name=name, shape=shape, dtype=dtype, type=type, lod_level=lod_level, stop_gradient=stop_gradient, trainable=trainable, persistable=False)
                if have_dist_attr:
                    var.is_distributed = is_distributed
            block._sync_with_cpp()
        recreate_not_persist_param_as_var(self._startup_program)
        recreate_not_persist_param_as_var(self._main_program)

    def _initialization_broadcast(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        this funtion is to ensure the initialization between dp group to be\n        identical when hybrid-dp is used, and the initialization of\n        not distributed param between mp group to be identical.\n        '
        if self.dp_degree <= 1 and self.mp_degree <= 1:
            return
        startup_block = self._startup_program.global_block()
        params = startup_block.all_parameters()
        params_name = []
        not_dist_param_name = set()
        for param in params:
            params_name.append(param.name)
            if not hasattr(param, 'is_distributed') or not param.is_distributed:
                not_dist_param_name.add(param.name)
        broadcast_params = set()
        for op in startup_block.ops:
            if op.type == 'c_broadcast':
                broadcast_params.add(op.desc.output_arg_names()[0])
        for param in params_name:
            if param in broadcast_params:
                continue
            rings = []
            if self.mp_degree > 1 and param in not_dist_param_name:
                rings.append(self.mp_ring_id)
            if self.dp_degree > 1:
                rings.append(self.dp_ring_id)
            for ring in rings:
                startup_block.append_op(type='c_broadcast', inputs={'X': param}, outputs={'Out': param}, attrs={'ring_id': ring, 'root': 0, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Forward})
        startup_block._sync_with_cpp()

    def create_persistable_gradients_and_insert_merge_ops(self, main_block, startup_block, insert_idx, grad_names, shard):
        if False:
            for i in range(10):
                print('nop')
        for grad_name in grad_names:
            assert get_grad_device(grad_name, shard) == shard.worker_idx, f'try to merge gradient not belong to current shard: [{grad_name}]'
            persistable_grad_name = grad_name + '@GradiantMerge'
            assert grad_name not in self._grad2merged_grad, 'grad [{}] already in grad2merged_grad, maybe you meet sharing weight case !'.format(grad_name)
            self._grad2merged_grad[grad_name] = persistable_grad_name
            grad_var = main_block.var(grad_name)
            gradient_merge_var = main_block.create_var(name=persistable_grad_name, shape=grad_var.shape, dtype=grad_var.dtype, persistable=True)
            startup_gradient_merge_var = startup_block.create_var(name=persistable_grad_name, shape=grad_var.shape, dtype=grad_var.dtype, persistable=True)
            main_block._insert_op_without_sync(insert_idx, type='elementwise_add', inputs={'X': grad_name, 'Y': gradient_merge_var}, outputs={'Out': gradient_merge_var}, attrs={'axis': -1, 'use_mkldnn': False, OP_ROLE_KEY: OpRole.Backward})
            startup_block.append_op(type='fill_constant', outputs={'Out': startup_gradient_merge_var}, attrs={'shape': grad_var.shape, 'dtype': grad_var.dtype, 'value': float(0)})
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _create_gm_cond(self, main_block):
        if False:
            print('Hello World!')
        acc_step_var = create_global_var(name='gradient_merge_acc_step', shape=[1], value=int(self._gradient_merge_acc_step), dtype='int32', persistable=True, force_cpu=True)
        zero_var = create_global_var(name='gradient_merge_zero', shape=[1], value=0, dtype='int32', persistable=True, force_cpu=True)
        current_step_var = create_global_var(name='gradient_merge_current_step', shape=[1], value=0, dtype='int32', persistable=True, force_cpu=True)
        cond_var = main_block.create_var(name='gradient_merge_cond', shape=[1], dtype='bool')
        with device_guard('cpu'):
            main_block.append_op(type='increment', inputs={'X': [current_step_var]}, outputs={'Out': [current_step_var]}, attrs={'step': float(1), OP_ROLE_KEY: OpRole.Optimize})
            main_block.append_op(type='elementwise_mod', inputs={'X': current_step_var, 'Y': acc_step_var}, outputs={'Out': current_step_var}, attrs={'axis': -1, OP_ROLE_KEY: OpRole.Optimize, 'use_mkldnn': False})
            main_block.append_op(type='equal', inputs={'X': current_step_var, 'Y': zero_var}, outputs={'Out': cond_var}, attrs={OP_ROLE_KEY: OpRole.Optimize})
        return cond_var

    def _true_apply_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        allreduce grad@gradientmerge in dp group\n        grad@gradientmerge / acc_step\n        re-create all optimize ops of origin main block and rename them\n            cast(backward)\n            amp\n            clip\n            opt\n        # fill constant grad@gradientmerge\n\n        '
        main_block = self._main_program.global_block()
        cur_block_idx = self._main_program.current_block_idx
        cur_block = self._main_program.current_block()
        self.cond_block = self._main_program.current_block()
        cur_block._set_forward_block_idx(cur_block_idx)
        if self.hybrid_dp:
            assert self.dp_ring_id >= 0, 'dp_ring_id should larger than 0 when in sharding&DP mode'
            for (grad, merged_grad) in self._grad2merged_grad.items():
                merged_grad_var = main_block.var(merged_grad)
                cur_block.append_op(type='c_allreduce_sum', inputs={'X': merged_grad_var}, outputs={'Out': merged_grad_var}, attrs={'ring_id': self.dp_ring_id, 'use_calc_stream': True, OP_ROLE_KEY: OpRole.Optimize})
        for (grad, merged_grad) in self._grad2merged_grad.items():
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(type='scale', inputs={'X': merged_grad_var}, outputs={'Out': merged_grad_var}, attrs={'scale': 1.0 / float(self._gradient_merge_acc_step), 'bias': 0.0, 'bias_after_scale': False, OP_ROLE_KEY: OpRole.Optimize})
        already_moved_var_names = []
        for op_desc in self.original_optimize_ops_desc:
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)
            for input_name in new_op_desc.input_arg_names():
                if input_name in self._grad2merged_grad:
                    new_op_desc._rename_input(input_name, self._grad2merged_grad[input_name])
            for output_name in new_op_desc.output_arg_names():
                if output_name in self._grad2merged_grad:
                    new_op_desc._rename_output(output_name, self._grad2merged_grad[output_name])
                if output_name not in already_moved_var_names and output_name not in self._grad2merged_grad.keys():
                    var_ = self._main_program.global_block().var(output_name)
                    if not var_.persistable:
                        name_ = var_.name
                        shape_ = var_.shape
                        type_ = var_.dtype
                        self._main_program.global_block()._remove_var(var_.name, sync=False)
                        self.cond_block.create_var(name=name_, shape=shape_, dtype=type_, persistable=False)
                        already_moved_var_names.append(name_)
        self._main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()
        for (grad, merged_grad) in self._grad2merged_grad.items():
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(type='fill_constant', outputs={'Out': merged_grad_var}, attrs={'shape': merged_grad_var.shape, 'dtype': merged_grad_var.dtype, 'value': float(0), OP_ROLE_KEY: OpRole.Optimize})

    def _sharding_gradient_merge(self):
        if False:
            i = 10
            return i + 15
        '\n        copy all optimize ops in origin main block\n        remove all optimize ops in origin main block\n        create cond block\n\n        '
        if self.gradient_merge_mode != 'sharding_gm' or self._gradient_merge_acc_step <= 1:
            return
        main_block = self._main_program.global_block()
        tmp_copy_block = self._main_program._create_block()
        self.original_optimize_ops_desc = []
        for (op_idx, op) in reversed(list(enumerate(main_block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                continue
            else:
                tmp_op_desc = tmp_copy_block.desc.append_op()
                tmp_op_desc.copy_from(op.desc)
                self.original_optimize_ops_desc.append(tmp_op_desc)
                main_block._remove_op(op_idx, sync=False)
        tmp_copy_block._sync_with_cpp()
        self.original_optimize_ops_desc = list(reversed(self.original_optimize_ops_desc))
        self._main_program._rollback()
        cond = self._create_gm_cond(main_block)
        cond_block = self._main_program._create_block()
        self._true_apply_gradient()
        self._main_program._rollback()
        step_scope = self._main_program.global_block().create_var(type=core.VarDesc.VarType.STEP_SCOPES)
        conditional_block_op = self._main_program.global_block().append_op(type='conditional_block', inputs={'Cond': cond, 'Input': []}, outputs={'Out': [], 'Scope': [step_scope]}, attrs={'sub_block': cond_block, 'is_scalar_condition': True})