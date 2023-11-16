"""
Communicator is used for async distribute training in distribute_transpiler mode.
It's a wrapper of a cpp class Communicator and should be used inside fleet API.
"""
import paddle
from paddle.distributed.ps.utils.public import DistributedMode
from paddle.framework import core
__all__ = []

class Communicator:

    def __init__(self, mode, kwargs=None, envs=None):
        if False:
            print('Hello World!')
        "\n        Communicator is used for async distribute training in distribute_transpiler mode.\n        It's a wrapper of a cpp class Communicator and should be used inside fleet API.\n\n        Args:\n            program(Program): the trainers program after transpile of distribute_transpiler.\n            It's used by communicator to extract the information to do communication.\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> prog = paddle.static.Program()\n                >>> comm = paddle.distributed.communicator.Communicator(prog)\n                >>> comm.start()\n                >>> comm.stop()\n        "
        if kwargs is None:
            if envs is None:
                envs = {}
        else:
            if mode == DistributedMode.SYNC:
                envs['pserver_endpoints'] = ','.join(kwargs['pserver_endpoints'])
            envs['trainers'] = str(kwargs['trainers'])
            envs['trainer_id'] = str(kwargs['trainer_id'])
            envs['need_global_step'] = str(kwargs['need_global_step'])
            envs['barrier_table_id'] = str(kwargs['barrier_table_id'])
        mode_str = None
        if mode == DistributedMode.SYNC:
            mode_str = 'SYNC'
        elif mode == DistributedMode.ASYNC:
            mode_str = 'ASYNC'
        elif mode == DistributedMode.HALF_ASYNC:
            mode_str = 'HALF_ASYNC'
        elif mode == DistributedMode.GEO:
            mode_str = 'GEO'
        self.mode = mode_str
        self.envs = envs
        self.communicator_ = None
        self.send_ctx_ = None
        self.recv_ctx_ = None

    def init_with_ctx(self, send_ctx, recv_ctx, proto_txt, unit64_hosts, scope=None):
        if False:
            return 10
        if scope is None:
            scope = paddle.static.global_scope()
        self.communicator_ = core.DistCommunicator(self.mode, proto_txt, unit64_hosts, send_ctx, recv_ctx, scope, self.envs)
        self.send_ctx_ = send_ctx
        self.recv_ctx_ = recv_ctx

    def create_client_to_client_connection(self, pserver_timeout_ms=500000, pserver_connect_timeout_ms=10000, max_retry=3):
        if False:
            print('Hello World!')
        self.communicator_.create_client_to_client_connection(pserver_timeout_ms, pserver_connect_timeout_ms, max_retry)

    def get_client_info(self):
        if False:
            while True:
                i = 10
        return self.communicator_.get_client_info()

    def set_clients(self, host_list):
        if False:
            return 10
        self.communicator_.set_clients(host_list)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start communicator. Should call before training process.\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> prog = paddle.static.Program()\n                >>> comm = paddle.distributed.communicator.Communicator(prog)\n                >>> comm.start()\n                >>> comm.stop()\n        '
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before start')
            return
        self.communicator_.start()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop communicator. Should call after training process.\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> prog = paddle.static.Program()\n                >>> comm = paddle.distributed.communicator.Communicator(prog)\n                >>> comm.start()\n                >>> comm.stop()\n        '
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before stop')
            return
        self.communicator_.stop()

    def is_running(self):
        if False:
            print('Hello World!')
        '\n        Get communicator is running or stop.\n\n        Returns:\n            bool\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> prog = paddle.static.Program()\n                >>> comm = paddle.distributed.communicator.Communicator(prog)\n                >>> comm.is_running()\n        '
        if self.communicator_ is None:
            print('you must call init_with_ctx first to init comm before stop')
            return
        self.communicator_.is_running()

    def recv(self):
        if False:
            i = 10
            return i + 15
        self.communicator_.recv()

    def init_params(self, context):
        if False:
            while True:
                i = 10
        self.communicator_.init_params(context)

    def pull_dense(self, context):
        if False:
            print('Hello World!')
        self.communicator_.pull_dense(context)

    def push_sparse_param(self, var_name, table_id=-1, scope=None):
        if False:
            return 10
        if scope is None:
            scope = paddle.static.global_scope()
        if not self.is_running():
            raise ValueError('Communicator should init first. Using fleet.init_worker() before push_sparse_param()')
        assert isinstance(var_name, str)
        assert isinstance(table_id, int)
        if table_id == -1:
            table_id = self.send_ctx_[var_name].table_id()
        self.communicator_.push_sparse_param(var_name, table_id, scope)

class FLCommunicator(Communicator):

    def __init__(self, ps_hosts, kwargs=None):
        if False:
            return 10
        mode = None
        super().__init__(mode, kwargs)
        send_ctx = {}
        dense_map = {}
        prototxt = ''
        self.mode = 'WITH_COORDINATOR'
        self.init_with_ctx(send_ctx, dense_map, prototxt, ps_hosts)

    def start_coordinator(self, self_endpoint, trainer_endpoints):
        if False:
            return 10
        if self.communicator_ is not None:
            self.communicator_.start_coordinator(self_endpoint, trainer_endpoints)

    def save_fl_strategy(self, mp):
        if False:
            print('Hello World!')
        if self.communicator_ is not None:
            self.communicator_.save_fl_strategy(mp)
        else:
            raise ValueError('self.communicator_ is null')

    def query_fl_clients_info(self):
        if False:
            print('Hello World!')
        info_mp = {}
        if self.communicator_ is not None:
            info_mp = self.communicator_.query_fl_clients_info()
        return info_mp

class LargeScaleKV:

    def __init__(self):
        if False:
            print('Hello World!')
        self.scale_kv = core.LargeScaleKV()

    def save(self, varname, dirname):
        if False:
            for i in range(10):
                print('nop')
        self.scale_kv.save(varname, dirname)

    def load(self, varname, dirname):
        if False:
            i = 10
            return i + 15
        self.scale_kv.load(varname, dirname)

    def size(self, varname):
        if False:
            return 10
        return self.scale_kv.size(varname)

class HeterClient:

    def __init__(self, endpoint, previous_endpoint, trainer_id):
        if False:
            i = 10
            return i + 15
        self.heter_client_ = core.HeterClient(endpoint, previous_endpoint, trainer_id)

    def stop(self):
        if False:
            print('Hello World!')
        self.heter_client_.stop()