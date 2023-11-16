"""Definition of trainers."""
import os
import sys
__all__ = []

class TrainerDesc:
    """
    Set proto from python to c++.
    Can be initialized from train_desc.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        "\n        self.proto_desc = data_feed_pb2.DataFeedDesc()\n        with open(proto_file, 'r') as f:\n            text_format.Parse(f.read(), self.proto_desc)\n        "
        cur_path = os.path.dirname(__file__)
        if cur_path not in sys.path:
            sys.path.append(cur_path)
        if cur_path + '/proto' not in sys.path:
            sys.path.append(cur_path + '/proto')
        from proto import trainer_desc_pb2
        self.proto_desc = trainer_desc_pb2.TrainerDesc()
        import multiprocessing as mp
        self.proto_desc.thread_num = mp.cpu_count()
        self._fleet_desc = None
        self._device_worker = None
        self._program = None
        self._infer = False

    def _set_heter_info(self, ret):
        if False:
            while True:
                i = 10
        if ret is None:
            return
        self.proto_desc.xpu_start_idx = ret[0]
        self.proto_desc.xpu_end_idx = ret[1]
        for i in ret[2]:
            self.proto_desc.xpu_send_list.append(i)
        for i in ret[3]:
            self.proto_desc.xpu_recv_list.append(i)

    def _set_fetch_var_and_info(self, fetch_vars, fetch_info, print_period):
        if False:
            print('Hello World!')
        fetch_info = list(fetch_info)
        for (i, v) in enumerate(fetch_vars):
            self.proto_desc.fetch_config.fetch_var_names.extend([v.name])
            self.proto_desc.fetch_config.fetch_var_str_format.extend([fetch_info[i]])
        self.proto_desc.fetch_config.print_period = print_period

    def _set_debug(self, debug):
        if False:
            while True:
                i = 10
        self.proto_desc.debug = debug

    def _set_thread(self, thread_num):
        if False:
            i = 10
            return i + 15
        self.proto_desc.thread_num = thread_num

    def _set_device_worker(self, device_worker):
        if False:
            while True:
                i = 10
        self._device_worker = device_worker

    def _set_infer(self, infer):
        if False:
            return 10
        self._infer = infer

    def _set_fleet_desc(self, fleet_desc):
        if False:
            i = 10
            return i + 15
        self._fleet_desc = fleet_desc
        from google.protobuf import text_format
        fleet_desc_str = text_format.MessageToString(fleet_desc)
        self.proto_desc.fleet_desc = fleet_desc_str

    def _gen_trainer_desc(self):
        if False:
            return 10
        pass

    def _set_program(self, program):
        if False:
            while True:
                i = 10
        self._program = program

    def _set_trainer_id(self, trainer_id):
        if False:
            return 10
        self.proto_desc.trainer_id = trainer_id

    def _set_trainers(self, trainers):
        if False:
            i = 10
            return i + 15
        for trainer_num in trainers:
            self.proto_desc.trainers.append(trainer_num)

    def _set_use_cvm(self, use_cvm=False):
        if False:
            while True:
                i = 10
        self.proto_desc.use_cvm = use_cvm

    def _set_no_cvm(self, no_cvm=False):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.no_cvm = no_cvm

    def _set_scale_sparse_grad_with_batch_size(self, scale_sparse_gradient_with_batch_size=True):
        if False:
            print('Hello World!')
        self.proto_desc.scale_sparse_gradient_with_batch_size = scale_sparse_gradient_with_batch_size

    def _set_scale_datanorm(self, scale_datanorm=-1):
        if False:
            print('Hello World!')
        self.proto_desc.scale_datanorm = scale_datanorm

    def _set_dump_slot(self, dump_slot):
        if False:
            print('Hello World!')
        self.proto_desc.dump_slot = dump_slot

    def _set_mpi_rank(self, mpi_rank):
        if False:
            return 10
        self.proto_desc.mpi_rank = mpi_rank

    def _set_mpi_size(self, mpi_size):
        if False:
            print('Hello World!')
        self.proto_desc.mpi_size = mpi_size

    def _set_dump_fields(self, dump_fields):
        if False:
            i = 10
            return i + 15
        for field in dump_fields:
            self.proto_desc.dump_fields.append(field)

    def _set_is_dump_in_simple_mode(self, is_dump_in_simple_mode):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.is_dump_in_simple_mode = is_dump_in_simple_mode

    def _set_dump_fields_path(self, path):
        if False:
            while True:
                i = 10
        self.proto_desc.dump_fields_path = path

    def _set_dump_file_num(self, dump_file_num):
        if False:
            return 10
        self.proto_desc.dump_file_num = dump_file_num

    def _set_user_define_dump_filename(self, user_define_dump_filename):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.user_define_dump_filename = user_define_dump_filename

    def _set_dump_converter(self, converter):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.dump_converter = converter

    def _set_enable_random_dump(self, enable_random_dump):
        if False:
            print('Hello World!')
        self.proto_desc.enable_random_dump = enable_random_dump

    def _set_dump_interval(self, dump_interval):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.dump_interval = dump_interval

    def _set_random_with_lineid(self, random_with_lineid):
        if False:
            return 10
        self.proto_desc.random_with_lineid = random_with_lineid

    def _set_dump_param(self, dump_param):
        if False:
            print('Hello World!')
        for param in dump_param:
            self.proto_desc.dump_param.append(param)

    def _set_worker_places(self, worker_places):
        if False:
            i = 10
            return i + 15
        for place in worker_places:
            self.proto_desc.worker_places.append(place)

    def _set_use_ps_gpu(self, use_ps_gpu=False):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.use_ps_gpu = use_ps_gpu

    def _set_thread_barrier(self, thread_barrier):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.thread_barrier = thread_barrier

    def _set_check_nan_var_names(self, check_nan_var_names):
        if False:
            return 10
        for var in check_nan_var_names:
            self.proto_desc.check_nan_var_names.append(var)

    def _set_loss_names(self, loss_names):
        if False:
            while True:
                i = 10
        for loss in loss_names:
            self.proto_desc.loss_names.append(loss)

    def _set_adjust_ins_weight(self, config_dict):
        if False:
            for i in range(10):
                print('nop')
        self.proto_desc.adjust_ins_weight_config.need_adjust = config_dict.get('need_adjust', False)
        self.proto_desc.adjust_ins_weight_config.nid_slot = config_dict.get('nid_slot', '')
        self.proto_desc.adjust_ins_weight_config.nid_adjw_threshold = config_dict.get('nid_adjw_threshold', 0.0)
        self.proto_desc.adjust_ins_weight_config.nid_adjw_ratio = config_dict.get('nid_adjw_ratio', 0.0)
        self.proto_desc.adjust_ins_weight_config.ins_weight_slot = config_dict.get('ins_weight_slot', '')

    def _set_copy_table_config(self, config_dict):
        if False:
            i = 10
            return i + 15
        config = self.proto_desc.copy_table_config
        config.need_copy = config_dict.get('need_copy', False)
        config.batch_num = config_dict.get('batch_num', 100)
        src_sparse_tables = config_dict.get('src_sparse_tables', [])
        if not isinstance(src_sparse_tables, list):
            src_sparse_tables = [src_sparse_tables]
        dest_sparse_tables = config_dict.get('dest_sparse_tables', [])
        if not isinstance(dest_sparse_tables, list):
            dest_sparse_tables = [dest_sparse_tables]
        if len(src_sparse_tables) != len(dest_sparse_tables):
            raise ValueError(f'len(src_sparse_tables) != len(dest_sparse_tables), {len(src_sparse_tables)} vs {len(dest_sparse_tables)}')
        for i in src_sparse_tables:
            config.src_sparse_tables.append(i)
        for i in dest_sparse_tables:
            config.dest_sparse_tables.append(i)
        src_dense_tables = config_dict.get('src_dense_tables', [])
        if not isinstance(src_dense_tables, list):
            src_dense_tables = [src_dense_tables]
        dest_dense_tables = config_dict.get('dest_dense_tables', [])
        if not isinstance(dest_dense_tables, list):
            dest_dense_tables = [dest_dense_tables]
        if len(src_dense_tables) != len(dest_dense_tables):
            raise ValueError(f'len(src_dense_tables) != len(dest_dense_tables), {len(src_dense_tables)} vs {len(dest_dense_tables)}')
        for i in src_dense_tables:
            config.src_dense_tables.append(i)
        for i in dest_dense_tables:
            config.dest_dense_tables.append(i)
        src_var_list = config_dict.get('src_var_list', [])
        if not isinstance(src_var_list, list):
            src_var_list = [src_var_list]
        dest_var_list = config_dict.get('dest_var_list', [])
        if not isinstance(dest_var_list, list):
            dest_var_list = [dest_var_list]
        if len(src_var_list) != len(dest_var_list):
            raise ValueError(f'len(src_var_list) != len(dest_var_list), {len(src_var_list)} vs {len(dest_var_list)}')
        for i in src_var_list:
            config.src_var_list.append(i)
        for i in dest_var_list:
            config.dest_var_list.append(i)
        dependency_map = config_dict.get('dependency_map', {})
        for key in dependency_map:
            m = config.table_denpendency_map.add()
            m.key = key
            values = dependency_map[key]
            if not isinstance(values, list):
                values = [values]
            if len(values) != 1:
                raise ValueError('dependency len %s != 1' % len(values))
            for value in values:
                m.values.append(value)
        config.dense_pull_after_copy = config_dict.get('dense_pull_after_copy', True)
        config.enable_dependency = config_dict.get('enable_dependency', False)
        config.sparse_copy_by_feasign = config_dict.get('sparse_copy_by_feasign', True)

    def _desc(self):
        if False:
            i = 10
            return i + 15
        return self.proto_desc.SerializeToString()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        from google.protobuf import text_format
        return text_format.MessageToString(self.proto_desc)

class MultiTrainer(TrainerDesc):
    """
    Implement of MultiTrainer.
    Can be init from TrainerDesc.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            print('Hello World!')
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            i = 10
            return i + 15
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'MultiTrainer'
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)

class DistMultiTrainer(TrainerDesc):
    """
    Implement of DistMultiTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            i = 10
            return i + 15
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            for i in range(10):
                print('nop')
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'DistMultiTrainer'
        if self._program is None:
            raise RuntimeError('None Program')
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)

class HeterXpuTrainer(TrainerDesc):
    """
    Implement of HeterXpuTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            while True:
                i = 10
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            i = 10
            return i + 15
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'HeterXpuTrainer'
        if self._program is None:
            raise RuntimeError('None Program')
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)

class PSGPUTrainer(TrainerDesc):
    """
    Implement of PSGPUTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            return 10
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            print('Hello World!')
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'PSGPUTrainer'
        if self._program is None:
            raise RuntimeError('None Program')
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)

class HeterPipelineTrainer(TrainerDesc):
    """
    Implement of HeterPipelineTrainer.
    It's for HeterPS Pipeline training.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            for i in range(10):
                print('nop')
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            return 10
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'HeterPipelineTrainer'
        if self._program is None:
            raise RuntimeError('None Program')
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)

class PipelineTrainer(TrainerDesc):
    """
    Implement of PipelineTrainer.
    It's for Pipeline.
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        pass

    def _set_program(self, program):
        if False:
            return 10
        super()._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        if False:
            print('Hello World!')
        super()._gen_trainer_desc()
        self.proto_desc.class_name = 'PipelineTrainer'
        if self._program is None:
            raise RuntimeError('None Program')
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)