import copy
import logging
import os
import time
from paddle.distributed.passes import PassManager, new_pass
from paddle.framework import get_flags
from paddle.static import append_backward, program_guard
from paddle.utils import unique_name
from ...utils.log_utils import get_logger
from ..random import init_auto_parallel_rng
from .partitioner import Partitioner
from .process_group import get_world_process_group
from .reshard import Resharder
from .utils import get_pp_stage, is_sequential_run, use_new_executor
NEW_IR_PASS = ['fused_gemm_epilogue_pass', 'fused_linear_param_grad_add_pass', 'fused_dropout_add_pass']

class Parallelizer:

    def __init__(self, mode, completer, dist_context):
        if False:
            print('Hello World!')
        self._mode = mode
        self._completer = completer
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._pass_context = self._dist_context.pass_context
        self._strategy = self._dist_context.strategy
        self._logger = get_logger(logging.INFO)

    @property
    def is_train(self):
        if False:
            i = 10
            return i + 15
        return self._mode == 'train'

    @property
    def is_test(self):
        if False:
            i = 10
            return i + 15
        return self._mode in ['eval', 'predict']

    def parallel_all(self, parameter_list=None):
        if False:
            return 10
        world_process_group = get_world_process_group()
        all_ranks = world_process_group.ranks
        for rank in all_ranks:
            self.parallel(rank, parameter_list)

    def parallel(self, rank, parameter_list=None):
        if False:
            while True:
                i = 10
        serial_main_program = self._dist_context.serial_main_program
        serial_startup_program = self._dist_context.serial_startup_program
        serial_optimizer = self._dist_context.serial_optimizer
        if self.is_train and serial_optimizer:
            serial_loss = self._dist_context.serial_loss
            params_grads = self._generate_backward(serial_main_program, serial_startup_program, serial_loss, parameter_list)
            time0 = time.time()
            (serial_main_program, serial_startup_program, params_grads) = self._apply_pre_optimization(serial_main_program, serial_startup_program, serial_loss, serial_optimizer, params_grads)
            self._logger.debug('within parallel apply_pre_optimization time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            partitioner = Partitioner(self._dist_context, rank)
            (dist_main_prog, dist_startup_prog, dist_params_grads) = partitioner.partition(serial_main_program, serial_startup_program, params_grads)
            init_auto_parallel_rng()
            self._logger.debug('within parallel partitioner time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            self._generate_optimizer(dist_main_prog, dist_startup_prog, serial_optimizer, dist_params_grads)
            self._logger.debug('within parallel optimizer time: {}, mode {}'.format(time.time() - time0, self._mode))
            resharder = Resharder(dist_main_prog, dist_startup_prog, rank, self._dist_context, dist_params_grads)
            resharder.reshard()
            self._logger.debug('within parallel reshard time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            self._apply_post_optimization(dist_main_prog, dist_startup_prog, rank, dist_params_grads)
            self._logger.debug('within parallel apply_post_optimization time: {}, mode {}'.format(time.time() - time0, self._mode))
        else:
            time0 = time.time()
            (serial_main_program, serial_startup_program, params_grads) = self._apply_pre_optimization(serial_main_program, serial_startup_program, None, None, [])
            self._logger.debug('within parallel apply_pre_optimization time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            partitioner = Partitioner(self._dist_context, rank)
            (dist_main_prog, dist_startup_prog, dist_params_grads) = partitioner.partition(serial_main_program, serial_startup_program, [])
            self._logger.debug('within parallel partitioner time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            micro_bsz = 1 if not self._strategy.pipeline.enable else self._strategy.pipeline.micro_batch_size
            resharder = Resharder(dist_main_prog, dist_startup_prog, rank, self._dist_context, [], micro_bsz)
            resharder.reshard()
            self._logger.debug('within parallel reshard time: {}, mode {}'.format(time.time() - time0, self._mode))
            time0 = time.time()
            self._apply_post_optimization(dist_main_prog, dist_startup_prog, rank, dist_params_grads)
            self._logger.debug('within parallel apply_post_optimization time: {}, mode {}'.format(time.time() - time0, self._mode))
        if self.is_test:
            pipeline_opt = dist_main_prog._pipeline_opt
            dist_main_prog = dist_main_prog.clone(for_test=True)
            dist_startup_prog = dist_startup_prog.clone(for_test=True)
            dist_main_prog._pipeline_opt = pipeline_opt
        self._dist_context.dist_main_programs[rank] = dist_main_prog
        self._dist_context.dist_startup_programs[rank] = dist_startup_prog

    def _generate_backward(self, main_program, startup_program, loss, parameter_list=None):
        if False:
            while True:
                i = 10
        with program_guard(main_program, startup_program):
            params_grads = append_backward(loss, parameter_list=parameter_list, distop_context=self._dist_context.dist_op_context)
        self._completer.complete_backward_annotation(main_program)
        self._dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _generate_optimizer(self, main_program, startup_program, optimizer, params_grads):
        if False:
            return 10
        learning_rate = optimizer._learning_rate
        optimizer = copy.deepcopy(optimizer)
        self._dist_context._serial_optimizer = optimizer
        self._dist_context._serial_optimizer._learning_rate = learning_rate
        optimizer._sorted = False
        with program_guard(main_program, startup_program):
            with unique_name.guard('opt_'):
                optimizer_ops = optimizer.apply_gradients(params_grads)
        self._completer.complete_update_annotation(main_program)
        return optimizer_ops

    def _apply_pre_optimization(self, main_program, startup_program, loss, optimizer, params_grads):
        if False:
            while True:
                i = 10
        if self._strategy is None:
            return
        if self._strategy.amp.enable:
            config = copy.deepcopy(self._strategy.amp.to_dict())
            config['dist_context'] = self._dist_context
            config['params_grads'] = params_grads
            config['loss'] = loss
            config['input_data'] = self._dist_context.serial_feed_vars['inputs'] + self._dist_context.serial_feed_vars['labels']
            self._logger.info('Applying AMP-{}-{} ...'.format(config['dtype'], config['level']))
            if config['level'] == 'o1':
                auto_parallel_amp_pass = new_pass('auto_parallel_amp', config)
                auto_parallel_amp_pass.apply([main_program], [startup_program], self._pass_context)
                loss = auto_parallel_amp_pass.get_loss()
            elif config['level'] in ['o2', 'o3']:
                config['base_opt'] = optimizer
                auto_parallel_fp16_pass = new_pass('auto_parallel_fp16', config)
                auto_parallel_fp16_pass.apply([main_program], [startup_program], self._pass_context)
                loss = auto_parallel_fp16_pass.get_loss()
            else:
                raise ValueError('AMP level should be one of o1, o2, o3')
        if self.is_train and self._strategy.qat.enable:
            config = copy.deepcopy(self._strategy.qat.to_dict())
            config['dist_context'] = self._dist_context
            config['params_grads'] = params_grads
            config['mode'] = self._mode
            config['loss'] = loss
            auto_parallel_quantization_pass = new_pass('auto_parallel_quantization', config)
            auto_parallel_quantization_pass.apply([main_program], [startup_program], self._pass_context)
            main_program = self._pass_context.get_attr('main_program')
            startup_program = self._pass_context.get_attr('startup_program')
            params_grads = self._pass_context.get_attr('params_grads')
            loss = self._pass_context.get_attr('loss')
        if self.is_train and self._strategy.recompute.enable:
            config = copy.deepcopy(self._strategy.recompute.to_dict())
            config['dist_context'] = self._dist_context
            config['no_grad_set'] = None
            config['loss'] = loss
            auto_parallel_recompute_pass = new_pass('auto_parallel_recompute', config)
            auto_parallel_recompute_pass.apply([main_program], [startup_program], self._pass_context)
        return (main_program, startup_program, params_grads)

    def _apply_post_optimization(self, main_program, startup_program, rank, params_grads):
        if False:
            return 10
        if self._strategy is None:
            return
        if self._strategy.sp_optimization.enable:
            config = copy.deepcopy(self._strategy.sp_optimization.to_dict())
            config['dist_context'] = self._dist_context
            config['global_rank'] = rank
            sp_pass = new_pass('auto_parallel_sequence_parallel_optimization', config)
            sp_pass.apply([main_program], [startup_program], self._pass_context)
        if self._strategy.dp_optimization.enable:
            config = copy.deepcopy(self._strategy.dp_optimization.to_dict())
            config['dist_context'] = self._dist_context
            config['global_rank'] = rank
            config['use_sharding'] = self._strategy.sharding.enable
            dp_pass = new_pass('auto_parallel_data_parallel_optimization', config)
            dp_pass.apply([main_program], [startup_program], self._pass_context)
        if self._strategy.sharding.enable:
            config = copy.deepcopy(self._strategy.sharding.to_dict())
            config['dist_context'] = self._dist_context
            config['params_grads'] = params_grads
            config['global_rank'] = rank
            auto_parallel_sharding_pass = new_pass('auto_parallel_sharding', config)
            auto_parallel_sharding_pass.apply([main_program], [startup_program], self._pass_context)
            params_grads = self._pass_context.get_attr('params_grads')
        if self._strategy.mp_optimization.allreduce_matmul_grad_overlapping:
            if int(os.getenv('CUDA_DEVICE_MAX_CONNECTIONS', '0')) != 1:
                self._logger.warning('You set mp_optimization.allreduce_matmul_grad_overlapping=True, but you did not set environment variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance.')
            allreduce_matmul_grad_overlapping_pass = new_pass('allreduce_matmul_grad_overlapping', {})
            allreduce_matmul_grad_overlapping_pass.apply([main_program], [startup_program], self._pass_context)
        if self.is_train:
            config = copy.deepcopy(self._strategy.sharding.to_dict())
            config['dist_context'] = self._dist_context
            config['params_grads'] = params_grads
            config['rank_id'] = rank
            auto_parallel_clip_pass = new_pass('auto_parallel_grad_clip', config)
            auto_parallel_clip_pass.apply([main_program], [startup_program], self._pass_context)
        if not is_sequential_run():
            config = {}
            config['dist_context'] = self._dist_context
            APSED_pass = new_pass('auto_parallel_supplement_explicit_dependencies', config)
            APSED_pass.apply([main_program], [startup_program], self._pass_context)
        if self.is_train and self._strategy.pipeline.enable:
            self._strategy.gradient_merge.enable = True
            self._strategy.gradient_merge.k_steps = self._strategy.pipeline.accumulate_steps
            self._strategy.gradient_merge.avg = True
        if self.is_train and self._strategy.gradient_merge.enable:
            config = copy.deepcopy(self._strategy.gradient_merge.to_dict())
            config['dist_context'] = self._dist_context
            config['params_grads'] = params_grads
            auto_parallel_gradient_merge_pass = new_pass('auto_parallel_gradient_merge_pass', config)
            auto_parallel_gradient_merge_pass.apply([main_program], [startup_program], self._pass_context)
        if self._strategy.pipeline.enable and (not use_new_executor()):
            config = copy.deepcopy(self._strategy.pipeline.to_dict())
            config['dist_context'] = self._dist_context
            auto_parallel_pipeline_pass = new_pass('auto_parallel_pipeline', config)
            auto_parallel_pipeline_pass.apply([main_program], [startup_program], self._pass_context)
        enable_ir = get_flags('FLAGS_enable_pir_in_executor')['FLAGS_enable_pir_in_executor']
        ir_pass_list = []
        if self.is_train and self._strategy.fused_passes.enable:
            if len(self._strategy.fused_passes.fused_passes_list) > 0:
                new_pass_list = []
                for p in self._strategy.fused_passes.fused_passes_list:
                    if p in NEW_IR_PASS and enable_ir:
                        ir_pass_list.append(p)
                    else:
                        new_pass_list.append(new_pass(p))
                pass_manager = PassManager(new_pass_list)
                pass_manager.apply([main_program], [startup_program])
        main_program._pass_opt = {}
        main_program._pass_opt['pass_list'] = ir_pass_list
        if self.is_train and self._strategy.pipeline.enable and use_new_executor():
            enable_send_recv_overlap = self._strategy.pipeline.enable_send_recv_overlap
            if enable_send_recv_overlap and int(os.getenv('CUDA_DEVICE_MAX_CONNECTIONS', '0')) != 1:
                self._logger.warning('You set pipeline.enable_send_recv_overlap=True, but you did not set environment variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance.')
            main_program._pipeline_opt = {}
            main_program._pipeline_opt['standalone_opt'] = {'enable_send_recv_overlap': enable_send_recv_overlap, 'schedule_mode': self._strategy.pipeline.schedule_mode, 'num_micro_batches': self._strategy.pipeline.accumulate_steps, 'pp_degree': len(self._dist_context.process_meshes), 'pp_stage': get_pp_stage(self._dist_context, rank)}