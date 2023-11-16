import logging
import os
import sys
import time
import paddle
from paddle.distributed import fleet
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset(inputs, config):
    if False:
        for i in range(10):
            print('nop')
    dataset = paddle.distributed.InMemoryDataset()
    dataset._set_use_ps_gpu(config.get('runner.use_gpu'))
    pipe_cmd = config.get('runner.pipe_command')
    dataset.init(use_var=inputs, pipe_command=pipe_cmd, batch_size=32, thread_num=int(config.get('runner.thread_num')), fs_name=config.get('runner.fs_name', ''), fs_ugi=config.get('runner.fs_ugi', ''))
    dataset.set_filelist(['train_data/sample_train.txt'])
    dataset.update_settings(parse_ins_id=config.get('runner.parse_ins_id', False), parse_content=config.get('runner.parse_content', False))
    return dataset

class Main:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.metrics = {}
        self.input_data = None
        self.reader = None
        self.exe = None
        self.model = None
        self.PSGPU = None
        self.train_result_dict = {}
        self.train_result_dict['speed'] = []
        self.train_result_dict['auc'] = []

    def run(self):
        if False:
            i = 10
            return i + 15
        from ps_dnn_trainer import YamlHelper
        yaml_helper = YamlHelper()
        config_yaml_path = 'config_gpubox.yaml'
        self.config = yaml_helper.load_yaml(config_yaml_path)
        os.environ['CPU_NUM'] = str(self.config.get('runner.thread_num'))
        fleet.init()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
        logger.info('Run Success, Exit.')
        logger.info('-' * 100)

    def network(self):
        if False:
            print('Hello World!')
        from ps_dnn_trainer import StaticModel, get_user_defined_strategy
        self.model = StaticModel(self.config)
        self.input_data = self.model.create_feeds()
        self.init_reader()
        self.metrics = self.model.net(self.input_data)
        self.inference_target_var = self.model.inference_target_var
        logger.info('cpu_num: {}'.format(os.getenv('CPU_NUM')))
        user_defined_strategy = get_user_defined_strategy(self.config)
        optimizer = paddle.optimizer.Adam(0.01, lazy_mode=True)
        optimizer = fleet.distributed_optimizer(optimizer, user_defined_strategy)
        optimizer.minimize(self.model._cost)
        logger.info('end network.....')

    def run_server(self):
        if False:
            while True:
                i = 10
        logger.info('Run Server Begin')
        fleet.init_server(self.config.get('runner.warmup_model_path'))
        fleet.run_server()

    def run_worker(self):
        if False:
            return 10
        logger.info('Run Worker Begin')
        use_cuda = int(self.config.get('runner.use_gpu'))
        use_auc = self.config.get('runner.use_auc', False)
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)
        '\n        with open("./{}_worker_main_program.prototxt".format(\n                fleet.worker_index()), \'w+\') as f:\n            f.write(str(paddle.static.default_main_program()))\n        with open("./{}_worker_startup_program.prototxt".format(\n                fleet.worker_index()), \'w+\') as f:\n            f.write(str(paddle.static.default_startup_program()))\n        '
        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()
        '\n        save_model_path = self.config.get("runner.model_save_path")\n        if save_model_path and (not os.path.exists(save_model_path)):\n            os.makedirs(save_model_path)\n        '
        reader_type = self.config.get('runner.reader_type', None)
        epochs = int(self.config.get('runner.epochs'))
        sync_mode = self.config.get('runner.sync_mode')
        gpus_env = os.getenv('FLAGS_selected_gpus')
        self.PSGPU = paddle.framework.core.PSGPU()
        gpuslot = [int(i) for i in range(1, self.model.sparse_inputs_slots)]
        gpu_mf_sizes = [self.model.sparse_feature_dim - 1] * (self.model.sparse_inputs_slots - 1)
        self.PSGPU.set_slot_vector(gpuslot)
        self.PSGPU.set_slot_dim_vector(gpu_mf_sizes)
        self.PSGPU.init_gpu_ps([int(s) for s in gpus_env.split(',')])
        gpu_num = len(gpus_env.split(','))
        opt_info = paddle.static.default_main_program()._fleet_opt
        if use_auc is True:
            opt_info['stat_var_names'] = [self.model.stat_pos.name, self.model.stat_neg.name]
        else:
            opt_info['stat_var_names'] = []
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.dataset_train_loop(epoch)
            epoch_time = time.time() - epoch_start_time
            self.PSGPU.end_pass()
            fleet.barrier_worker()
            self.reader.release_memory()
            logger.info(f'finish {epoch} epoch training....')
        self.PSGPU.finalize()

    def init_reader(self):
        if False:
            print('Hello World!')
        if fleet.is_server():
            return
        self.reader = get_dataset(self.input_data, self.config)

    def dataset_train_loop(self, epoch):
        if False:
            while True:
                i = 10
        start_time = time.time()
        self.reader.load_into_memory()
        print(f'self.reader.load_into_memory cost :{time.time() - start_time} seconds')
        begin_pass_time = time.time()
        self.PSGPU.begin_pass()
        print(f'begin_pass cost:{time.time() - begin_pass_time} seconds')
        logger.info(f'Epoch: {epoch}, Running Dataset Begin.')
        fetch_info = [f'Epoch {epoch} Var {var_name}' for var_name in self.metrics]
        fetch_vars = [var for (_, var) in self.metrics.items()]
        print_step = int(self.config.get('runner.print_interval'))
        self.exe.train_from_dataset(program=paddle.static.default_main_program(), dataset=self.reader, debug=self.config.get('runner.dataset_debug'))
if __name__ == '__main__':
    paddle.enable_static()
    benchmark_main = Main()
    benchmark_main.run()