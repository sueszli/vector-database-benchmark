"""
high level unit test for distribute fleet.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
from paddle.distributed.fleet.utils.ps_util import DistributedInfer
paddle.enable_static()
__all__ = ['FleetDistRunnerBase', 'TestFleetBase', 'runtime_main']
RUN_STEP = 5
LEARNING_RATE = 0.01
DIST_UT_PORT = 0

class FleetDistRunnerBase:
    """
    run_pserver,run_trainer : after init role, using transpiler split program
    net : implment by child class, the network of model
    do training : exe run program
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._exe = None

    def build_role(self, args):
        if False:
            i = 10
            return i + 15
        if args.role.upper() == 'PSERVER':
            role = role_maker.UserDefinedRoleMaker(is_collective=False, init_gloo=False, path=args.gloo_path, current_id=args.current_id, role=role_maker.Role.SERVER, worker_endpoints=args.trainer_endpoints.split(','), server_endpoints=args.endpoints.split(','))
        else:
            role = role_maker.UserDefinedRoleMaker(is_collective=False, init_gloo=False, path=args.gloo_path, current_id=args.current_id, role=role_maker.Role.WORKER, worker_endpoints=args.trainer_endpoints.split(','), server_endpoints=args.endpoints.split(','))
        self.role = role
        return role

    def build_strategy(self, args):
        if False:
            while True:
                i = 10
        if args.mode == 'sync':
            self.strategy = paddle.distributed.fleet.DistributedStrategy()
            self.strategy.a_sync = False
        elif args.mode == 'async':
            self.strategy = paddle.distributed.fleet.DistributedStrategy()
            self.strategy.a_sync = True
        elif args.mode == 'geo':
            self.strategy = paddle.distributed.fleet.DistributedStrategy()
            self.strategy.a_sync = True
            self.strategy.a_sync_configs = {'k_steps': args.geo_sgd_need_push_nums}
        elif args.mode == 'auto':
            self.strategy = paddle.distributed.fleet.DistributedStrategy()
            self.strategy.auto = True
        self.dump_param = os.getenv('dump_param', '').split(',')
        self.dump_fields = os.getenv('dump_fields', '').split(',')
        self.dump_fields_path = os.getenv('dump_fields_path', '')
        debug = int(os.getenv('Debug', '0'))
        if False:
            self.strategy.set_debug_opt({'dump_param': self.dump_param, 'dump_fields': self.dump_fields, 'dump_fields_path': self.dump_fields_path})
        return self.strategy

    def build_optimizer(self, avg_cost, strategy):
        if False:
            for i in range(10):
                print('nop')
        use_grad_clip = int(os.getenv('GRAD_CLIP', 0))
        grad_clip = None
        if use_grad_clip:
            if use_grad_clip == 1:
                grad_clip = paddle.nn.ClipGradByValue(min=-5.0, max=5.0)
            elif use_grad_clip == 2:
                grad_clip = paddle.nn.ClipGradByNorm(2.0)
            elif use_grad_clip == 3:
                grad_clip = paddle.nn.ClipGradByGlobalNorm(2.0)
        use_decay = int(os.getenv('USE_DECAY', '0'))
        if use_decay:
            scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=LEARNING_RATE, gamma=0.999, verbose=True)
            optimizer = paddle.optimizer.SGD(scheduler, grad_clip=grad_clip)
            '\n            # learning rate decay method before 2.0\n            optimizer = base.optimizer.SGD(\n                learning_rate=base.layers.exponential_decay(\n                    learning_rate=LEARNING_RATE,\n                    decay_steps=500,\n                    decay_rate=0.969,\n                    staircase=True))\n            '
        else:
            optimizer = paddle.optimizer.SGD(LEARNING_RATE, grad_clip=grad_clip)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

    def run_pserver(self, args):
        if False:
            while True:
                i = 10
        fleet.init_server()
        fleet.run_server()

    def run_dataset_trainer(self, args):
        if False:
            for i in range(10):
                print('nop')
        out = self.do_dataset_training(fleet)

    def run_pyreader_trainer(self, args):
        if False:
            return 10
        out = self.do_pyreader_training(fleet)

    def net(self, args, batch_size=4, lr=0.01):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('get_model should be implemented by child classes.')

    def get_executor(self):
        if False:
            for i in range(10):
                print('nop')
        if self._exe is None:
            device_env = os.getenv('DEVICE', 'cpu')
            if device_env == 'cpu':
                device = base.CPUPlace()
            elif device_env == 'gpu':
                device = base.CUDAPlace(0)
            self._exe = base.Executor(device)
        return self._exe

    def do_dataset_training(self, fleet):
        if False:
            print('Hello World!')
        raise NotImplementedError('do_dataset_training should be implemented by child classes.')

    def do_pyreader_training(self, fleet):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('do_pyreader_training should be implemented by child classes.')

    def do_distributed_testing(self, fleet):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('do_distributed_testing should be implemented by child classes.')

class TestFleetBase(unittest.TestCase):
    """
    start_pserver,start_trainer : add start cmd to test
    run_cluster : using multi process to test distribute program
    """

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('tests should have _setup_config implemented')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        t = time.time() - self.startTime
        print(f'{self.__class__.__name__}: {t:.3f}')

    def setUp(self):
        if False:
            print('Hello World!')
        self.startTime = time.time()
        self._mode = 'sync'
        self._reader = 'pyreader'
        self._trainers = 2
        self._pservers = 2
        self._need_test = 0
        self._model_dir = ''
        self._port_set = set()
        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv('PADDLE_DIST_UT_PORT'):
            DIST_UT_PORT = int(os.getenv('PADDLE_DIST_UT_PORT'))
        if DIST_UT_PORT:
            print('set begin_port:', DIST_UT_PORT)
            self._ps_endpoints = '127.0.0.1:{},127.0.0.1:{}'.format(DIST_UT_PORT, DIST_UT_PORT + 1)
            self._tr_endpoints = '127.0.0.1:{},127.0.0.1:{}'.format(DIST_UT_PORT + 2, DIST_UT_PORT + 3)
            DIST_UT_PORT += 4
        else:
            self._ps_endpoints = '127.0.0.1:{},127.0.0.1:{}'.format(self._find_free_port(), self._find_free_port())
            self._tr_endpoints = '127.0.0.1:{},127.0.0.1:{}'.format(self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable
        self._geo_sgd_need_push_nums = 5
        self._grad_clip_mode = 0
        self._setup_config()

    def _find_free_port(self):
        if False:
            print('Hello World!')

        def __free_port():
            if False:
                for i in range(10):
                    print('nop')
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def _start_pserver(self, cmd, required_envs):
        if False:
            while True:
                i = 10
        (ps0_cmd, ps1_cmd) = (cmd.format(0), cmd.format(1))
        log_dirname = required_envs.get('LOG_DIRNAME', tempfile.gettempdir())
        log_prename = required_envs.get('LOG_PREFIX', '')
        if log_dirname:
            log_prename += '_'
        ps0_err_log = os.path.join(log_dirname, log_prename + 'ps0_stderr.log')
        ps1_err_log = os.path.join(log_dirname, log_prename + 'ps1_stderr.log')
        ps0_out_log = os.path.join(log_dirname, log_prename + 'ps0_stdout.log')
        ps1_out_log = os.path.join(log_dirname, log_prename + 'ps1_stdout.log')
        ps0_err = open(ps0_err_log, 'wb+')
        ps1_err = open(ps1_err_log, 'wb+')
        ps0_out = open(ps0_out_log, 'wb+')
        ps1_out = open(ps1_out_log, 'wb+')
        ps0_proc = subprocess.Popen(ps0_cmd.strip().split(' '), stdout=ps0_out, stderr=ps0_err, env=required_envs)
        ps1_proc = subprocess.Popen(ps1_cmd.strip().split(' '), stdout=ps1_out, stderr=ps1_err, env=required_envs)
        return ((ps0_proc, ps0_out, ps0_err, ps0_out_log, ps0_err_log), (ps1_proc, ps1_out, ps1_err, ps1_out_log, ps1_err_log))

    def _start_trainer(self, cmd, required_envs):
        if False:
            while True:
                i = 10
        (tr0_cmd, tr1_cmd) = (cmd.format(0), cmd.format(1))
        log_dirname = required_envs.get('LOG_DIRNAME', tempfile.gettempdir())
        log_prename = required_envs.get('LOG_PREFIX', '')
        if log_dirname:
            log_prename += '_'
        tr0_err_log = os.path.join(log_dirname, log_prename + 'tr0_stderr.log')
        tr1_err_log = os.path.join(log_dirname, log_prename + 'tr1_stderr.log')
        tr0_out_log = os.path.join(log_dirname, log_prename + 'tr0_stdout.log')
        tr1_out_log = os.path.join(log_dirname, log_prename + 'tr1_stdout.log')
        tr0_err = open(tr0_err_log, 'wb+')
        tr1_err = open(tr1_err_log, 'wb+')
        tr0_out = open(tr0_out_log, 'wb+')
        tr1_out = open(tr1_out_log, 'wb+')
        tr0_proc = subprocess.Popen(tr0_cmd.strip().split(' '), stdout=tr0_out, stderr=tr0_err, env=required_envs)
        tr1_proc = subprocess.Popen(tr1_cmd.strip().split(' '), stdout=tr1_out, stderr=tr1_err, env=required_envs)
        return ((tr0_proc, tr0_out, tr0_err, tr0_out_log, tr0_err_log), (tr1_proc, tr1_out, tr1_err, tr1_out_log, tr1_err_log))

    def _run_cluster(self, model, envs):
        if False:
            for i in range(10):
                print('nop')
        env = {'GRAD_CLIP': str(self._grad_clip_mode), 'WITH_DISTRIBUTE': 'ON'}
        python_path = self._python_interp
        gloo_path = tempfile.mkdtemp()
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            python_path += ' -m coverage run --branch -p'
        env.update(envs)
        tr_cmd = '{} {} --role trainer --endpoints {} --trainer_endpoints {} --current_id {{}} --trainers {} --mode {} --geo_sgd_need_push_nums {} --reader {} --gloo_path {} --test {}'.format(python_path, model, self._ps_endpoints, self._tr_endpoints, self._trainers, self._mode, self._geo_sgd_need_push_nums, self._reader, gloo_path, self._need_test)
        ps_cmd = '{} {} --role pserver --endpoints {} --trainer_endpoints {} --current_id {{}} --trainers {} --mode {} --geo_sgd_need_push_nums {} --reader {} --gloo_path {} --test {}'.format(python_path, model, self._ps_endpoints, self._tr_endpoints, self._trainers, self._mode, self._geo_sgd_need_push_nums, self._reader, gloo_path, self._need_test)
        if self._model_dir:
            tr_cmd += f' --model_dir {self._model_dir}'
            ps_cmd += f' --model_dir {self._model_dir}'
        (ps0, ps1) = self._start_pserver(ps_cmd, env)
        (tr0, tr1) = self._start_trainer(tr_cmd, env)
        (ps0_proc, ps0_out, ps0_err, ps0_out_log, ps0_err_log) = ps0
        (ps1_proc, ps1_out, ps1_err, ps1_out_log, ps1_err_log) = ps1
        (tr0_proc, tr0_out, tr0_err, tr0_out_log, tr0_err_log) = tr0
        (tr1_proc, tr1_out, tr1_err, tr1_out_log, tr1_err_log) = tr1
        time_out = 60
        cur_time = 0
        while True:
            stat0 = tr0_proc.poll()
            stat1 = tr1_proc.poll()
            if stat0 is not None and stat1 is not None:
                break
            else:
                time.sleep(0.5)
                cur_time += 0.5
            if cur_time >= time_out:
                tr0_proc.terminate()
                tr1_proc.terminate()
                tr0_proc.wait()
                tr1_proc.wait()
                break
        tr0_ret = tr0_proc.returncode
        tr1_ret = tr1_proc.returncode
        ps0_proc.kill()
        ps1_proc.kill()
        ps0_proc.wait()
        ps1_proc.wait()

        def is_listen_failed(logx):
            if False:
                print('Hello World!')
            is_lf = False
            listen_rgx = 'Fail to listen'
            with open(logx, 'r') as rb:
                for line in rb.readlines():
                    if listen_rgx in line:
                        is_lf = True
                        break
            return is_lf

        def catlog(logx):
            if False:
                return 10
            basename = os.path.basename(logx)
            print(f'\n================== Error {basename} begin =====================')
            if not os.path.isfile(logx):
                raise FileNotFoundError(f'{logx} is not a file')
            os.system(f'cat {logx}')
            print(f'================== Error {basename} end =====================\n')
        if tr0_ret != 0 or tr1_ret != 0:
            if is_listen_failed(ps0_err_log) or is_listen_failed(ps1_err_log):
                print('find parameter server port bind failed, skip the error')
                (tr0_ret, tr1_ret) = (0, 0)
            else:
                for (out, err) in [(ps0_out_log, ps0_err_log), (ps1_out_log, ps1_err_log), (tr0_out_log, tr0_err_log), (tr1_out_log, tr1_err_log)]:
                    catlog(out)
                    catlog(err)
        for pipe in [tr0_err, tr0_out, tr1_err, tr1_out, ps0_err, ps0_out, ps1_err, ps1_out]:
            pipe.close()
        shutil.rmtree(gloo_path)
        self.assertEqual(tr0_ret, 0, 'something wrong in tr0, please check')
        self.assertEqual(tr1_ret, 0, 'something wrong in tr1, please check')
        return (0, 0)

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            print('Hello World!')
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '5000', 'http_proxy': ''}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

def runtime_main(test_class):
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Run Fleet test.')
    parser.add_argument('--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default='')
    parser.add_argument('--trainer_endpoints', type=str, required=False, default='')
    parser.add_argument('--gloo_path', type=str, required=False, default='')
    parser.add_argument('--current_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--mode', type=str, required=False, default='geo')
    parser.add_argument('--geo_sgd_need_push_nums', type=int, required=False, default=2)
    parser.add_argument('--reader', type=str, required=False, default='dataset')
    parser.add_argument('--test', type=int, required=False, default=0)
    parser.add_argument('--model_dir', type=str, required=False, default='')
    args = parser.parse_args()
    model = test_class()
    role = model.build_role(args)
    if args.test and args.model_dir != '':
        avg_cost = model.net(args, is_train=False)
        dist_infer = DistributedInfer()
        dist_infer.init_distributed_infer_env(exe=model.get_executor(), loss=model.avg_cost, role_maker=role, dirname=args.model_dir)
        if fleet.is_worker():
            with paddle.static.program_guard(main_program=dist_infer.get_dist_infer_program()):
                model.do_distributed_testing(fleet)
                fleet.stop_worker()
            return
        if fleet.is_server():
            return
    fleet.init(role)
    strategy = model.build_strategy(args)
    avg_cost = model.net(args)
    model.build_optimizer(avg_cost, strategy)
    if args.role == 'pserver':
        model.run_pserver(args)
    else:
        if args.reader == 'dataset':
            model.run_dataset_trainer(args)
        else:
            model.run_pyreader_trainer(args)
        if args.test:
            test_origin_program = paddle.static.Program()
            test_startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program=test_origin_program, startup_program=test_startup_program):
                with paddle.utils.unique_name.guard():
                    avg_cost = model.net(args, is_train=False)
            dist_infer = DistributedInfer(main_program=test_origin_program, startup_program=test_startup_program)
            with paddle.static.program_guard(main_program=dist_infer.get_dist_infer_program()):
                model.do_distributed_testing(fleet)
        fleet.stop_worker()