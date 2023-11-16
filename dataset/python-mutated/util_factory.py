"""Fleet Utils."""
'distributed operations'
'basic collective operations in python'
'remote file system'
import os
import re
import subprocess
from collections import OrderedDict
import numpy as np
from google.protobuf import text_format
import paddle
from paddle import framework
from paddle.base import core
from paddle.base.proto import framework_pb2
from paddle.static import Program
from ..utils.fs import FS
from .graphviz import GraphPreviewGenerator
__all__ = []

class UtilFactory:

    def _create_util(self, context=None):
        if False:
            print('Hello World!')
        util = UtilBase()
        if context is not None and 'valid_strategy' in context:
            util._set_strategy(context['valid_strategy'])
        if context is not None and 'role_maker' in context:
            util._set_role_maker(context['role_maker'])
        return util

class UtilBase:

    def __init__(self):
        if False:
            return 10
        self.role_maker = None
        self.dist_strategy = None

    def _set_strategy(self, dist_strategy):
        if False:
            for i in range(10):
                print('nop')
        self.dist_strategy = dist_strategy

    def _set_role_maker(self, role_maker):
        if False:
            while True:
                i = 10
        self.role_maker = role_maker

    def _set_file_system(self, fs_client):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(fs_client, FS), 'fs_client must be the instance of paddle.distributed.fleet.utils.FS'
        self.fs_client = fs_client

    def all_reduce(self, input, mode='sum', comm_world='worker'):
        if False:
            return 10
        '\n        All reduce `input` between specified collection. This is a distributed API.\n\n        Args:\n            input (list|tuple|numpy.array): The input variable to do all_reduce between specified collection.\n            mode (str): "sum" or "min" or "max".\n            comm_world (str, optional): Collection used to execute all_reduce operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .\n\n        Returns:\n            output(Numpy.array|None): A numpy array with the same shape as the `input` .\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n                >>> # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .\n                >>> import paddle.distributed.fleet as fleet\n                >>> from paddle.distributed.fleet import PaddleCloudRoleMaker\n                >>> import sys\n                >>> import numpy as np\n                >>> import os\n\n                >>> os.environ["PADDLE_WITH_GLOO"] = "2"\n\n                >>> def train():\n                ...     role = PaddleCloudRoleMaker(\n                ...         is_collective=False,\n                ...         init_gloo=True,\n                ...         path="./tmp_gloo")\n                ...     fleet.init(role)\n                ...\n                ...     if fleet.is_server():\n                ...         input = np.array([1, 2])\n                ...         output = fleet.util.all_reduce(input, "sum", "server")\n                ...         print(output) # [2, 4]\n                ...     elif fleet.is_worker():\n                ...         input = np.array([3, 4])\n                ...         output = fleet.util.all_reduce(input, "sum", "worker")\n                ...         print(output) # [6, 8]\n                ...     output = fleet.util.all_reduce(input, "sum", "all")\n                ...     print(output) # [8, 12]\n\n                >>> if __name__ == "__main__":\n                ...     train()\n        '
        if isinstance(input, tuple):
            input = list(input)
        return self.role_maker._all_reduce(input, mode, comm_world)

    def barrier(self, comm_world='worker'):
        if False:
            i = 10
            return i + 15
        '\n        Barrier between specified collection.\n\n        Args:\n            comm_world (str, optional): Collection used to execute barrier operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n                >>> # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .\n                >>> import paddle.distributed.fleet as fleet\n                >>> from paddle.distributed.fleet import PaddleCloudRoleMaker\n                >>> import sys\n                >>> import os\n\n                >>> os.environ["PADDLE_WITH_GLOO"] = "2"\n\n                >>> def train():\n                ...     role = PaddleCloudRoleMaker(\n                ...         is_collective=False,\n                ...         init_gloo=True,\n                ...         path="./tmp_gloo")\n                ...     fleet.init(role)\n                ...\n                ...     if fleet.is_server():\n                ...         fleet.util.barrier("server")\n                ...         print("all server arrive here") # all server arrive here\n                ...     elif fleet.is_worker():\n                ...         fleet.util.barrier("worker")\n                ...         print("all server arrive here") # all server arrive here\n                ...     fleet.util.barrier("all")\n                ...     print("all servers and workers arrive here") #all servers and workers arrive here\n\n                >>> if __name__ == "__main__":\n                ...     train()\n        '
        self.role_maker._barrier(comm_world)

    def all_gather(self, input, comm_world='worker'):
        if False:
            while True:
                i = 10
        '\n        All gather `input` between specified collection.\n\n        Args:\n            input (Int|Float): The input variable to do all_gather between specified collection.\n            comm_world (str, optional): Collection used to execute all_reduce operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .\n\n        Returns:\n            output (List): A list of gathered values.\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n                >>> # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .\n                >>> import paddle.distributed.fleet as fleet\n                >>> from paddle.distributed.fleet import PaddleCloudRoleMaker\n                >>> import sys\n                >>> import os\n\n                >>> os.environ["PADDLE_WITH_GLOO"] = "2"\n\n                >>> def train():\n                ...     role = PaddleCloudRoleMaker(\n                ...         is_collective=False,\n                ...         init_gloo=True,\n                ...         path="./tmp_gloo")\n                ...     fleet.init(role)\n                ...\n                ...     if fleet.is_server():\n                ...         input = fleet.server_index()\n                ...         output = fleet.util.all_gather(input, "server")\n                ...         print(output) # [0, 1]\n                ...     elif fleet.is_worker():\n                ...         input = fleet.worker_index()\n                ...         output = fleet.util.all_gather(input, "worker")\n                ...         print(output) # [0, 1]\n                ...     output = fleet.util.all_gather(input, "all")\n                ...     print(output) # [0, 1, 0, 1]\n\n                >>> if __name__ == "__main__":\n                ...     train()\n        '
        return self.role_maker._all_gather(input, comm_world)

    def _broadcast(self):
        if False:
            i = 10
            return i + 15
        pass

    def _scatter(self):
        if False:
            while True:
                i = 10
        pass

    def get_heter_file_shard(self, files):
        if False:
            i = 10
            return i + 15
        if not isinstance(files, list):
            raise TypeError('files should be a list of file need to be read.')
        trainers = self.role_maker._worker_num()
        trainer_id = self.role_maker._worker_index() - trainers
        remainder = len(files) % trainers
        blocksize = int(len(files) / trainers)
        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1
        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]
        return trainer_files[trainer_id]

    def get_file_shard(self, files):
        if False:
            print('Hello World!')
        '\n        Split files before distributed training, and return filelist assigned to the current trainer.\n\n        .. code-block:: text\n\n            example 1: files is [a, b, c ,d, e]  and trainer_num = 2, then trainer\n                    0 gets [a, b, c] and trainer 1 gets [d, e].\n            example 2: files is [a, b], and trainer_num = 3, then trainer 0 gets\n                    [a], trainer 1 gets [b],  trainer 2 gets []\n\n        Args:\n            files(list): File list need to be read.\n\n        Returns:\n            List: Files belong to this worker.\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n                >>> import paddle.distributed.fleet as fleet\n                >>> from paddle.distributed.fleet import UserDefinedRoleMaker\n\n                >>> role = UserDefinedRoleMaker(\n                ...     is_collective=False,\n                ...     init_gloo=False,\n                ...     current_id=0,\n                ...     role=fleet.Role.WORKER,\n                ...     worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],\n                ...     server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])\n                >>> fleet.init(role)\n\n                >>> files = fleet.util.get_file_shard(["file1", "file2", "file3"])\n                >>> print(files)\n                ["file1", "file2"]\n        '
        if not isinstance(files, list):
            raise TypeError('files should be a list of file need to be read.')
        trainer_id = self.role_maker._worker_index()
        trainers = self.role_maker._worker_num()
        remainder = len(files) % trainers
        blocksize = int(len(files) / trainers)
        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1
        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]
        return trainer_files[trainer_id]

    def print_on_rank(self, message, rank_id):
        if False:
            while True:
                i = 10
        '\n        Woker of rank `rank_id` print some message.\n\n        Args:\n            message(str): Log to be printed.\n            rank_id(int): trainer id.\n\n        Examples:\n\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n                >>> import paddle.distributed.fleet as fleet\n                >>> from paddle.distributed.fleet import UserDefinedRoleMaker\n\n                >>> role = UserDefinedRoleMaker(\n                ...     is_collective=False,\n                ...     init_gloo=False,\n                ...     current_id=0,\n                ...     role=fleet.Role.WORKER,\n                ...     worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],\n                ...     server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])\n                >>> fleet.init(role)\n\n                >>> fleet.util.print_on_rank("I\'m worker 0", 0)\n                I\'m worker 0\n        '
        if self.role_maker._worker_index() != rank_id:
            return
        print(message)

    def _save_program(self, program, model_filename='__model__', is_text=False):
        if False:
            return 10
        if is_text:
            with open(model_filename, 'w') as f:
                f.write(str(program))
        else:
            with open(model_filename, 'wb') as f:
                f.write(program.desc.serialize_to_string())

    def _load_program(self, path, is_text):
        if False:
            i = 10
            return i + 15

        def load_program_binary(path):
            if False:
                while True:
                    i = 10
            'load program from binary string file'
            with open(path, 'rb') as f:
                program_desc_str = f.read()
            return Program.parse_from_string(program_desc_str)

        def load_program_text(path):
            if False:
                return 10
            'load program from human-readable text file'
            with open(path, 'r') as f:
                program_desc_text = f.read()
            prog_desc = framework_pb2.ProgramDesc()
            text_format.Merge(program_desc_text, prog_desc)
            return Program.parse_from_string(prog_desc.SerializeToString())
        if is_text:
            return load_program_text(path)
        else:
            return load_program_binary(path)

    def _program_type_trans(self, prog_dir, prog_fn, is_text):
        if False:
            return 10
        prog = self._load_program(os.path.join(prog_dir, prog_fn), is_text)
        prog_out_fn = prog_fn + '.bin' if is_text else prog_fn + '.pbtxt'
        self._save_program(prog, os.path.join(prog_dir, prog_out_fn), 1 - is_text)
        return prog_out_fn

    def _visualize_graphviz(self, program, output_dir, output_filename):
        if False:
            i = 10
            return i + 15
        block = program.global_block()
        dot_path = os.path.join(output_dir, output_filename + '.dot')
        pdf_path = os.path.join(output_dir, output_filename + '.pdf')
        draw_block_graphviz(block, path=dot_path)
        cmd = ['dot', '-Tpdf', dot_path, '-o', pdf_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()

    def _proto_check(self, config):
        if False:
            while True:
                i = 10
        train_prog = self._load_program(config.train_prog_path, config.is_text_train_program)
        pruned_prog = self._load_program(config.pruned_prog_path, config.is_text_pruned_program)
        is_match = True
        pruned_vars = [(v.name, v) for v in pruned_prog.list_vars() if paddle.static.io.is_persistable(v)]
        pruned_vars = OrderedDict(pruned_vars)
        pruned_vars_name = list(pruned_vars)
        print(f'persistable vars in pruned program: {pruned_vars_name}')
        feed_fetch_type_list = [core.VarDesc.VarType.FEED_MINIBATCH, core.VarDesc.VarType.FETCH_LIST]
        for var_name in pruned_vars:
            var = pruned_vars[var_name]
            if var.type in feed_fetch_type_list:
                break
            try:
                train_prog_var = train_prog.global_block().var(var_name)
            except ValueError as e:
                print("Not find variable '%s' in train program. please check pruning." % var_name)
                is_match = False
                continue
            if var.shape != train_prog_var.shape or var.dtype != train_prog_var.dtype:
                print('variable: {} not match. in pruned program shape: {} dtype:{}, in train program shape: {} dtype: {}'.format(var_name, var.shape, var.dtype, train_prog_var.shape, train_prog_var.dtype))
                is_match = False
        return is_match

    def _params_check(self, config):
        if False:
            for i in range(10):
                print('nop')

        def feed_gen(batch_size, feeded_vars_dims, feeded_vars_filelist):
            if False:
                for i in range(10):
                    print('nop')

            def reader(batch_size, fn, dim):
                if False:
                    return 10
                data = []
                if isinstance(dim, (list, tuple)):
                    shape = list(dim)
                    _temp = 1
                    for x in dim:
                        _temp = _temp * x
                    dim = _temp
                else:
                    shape = [dim]
                shape = [batch_size] + shape
                dim = dim * batch_size
                for line in open(fn, 'r'):
                    fields = line.strip().split(' ')
                    fields = [float(d) for d in fields]
                    while len(fields) >= dim:
                        tmp = fields[:dim]
                        fields = fields[dim:]
                        data.append(np.array(tmp).reshape(shape))
                return data
            batch_feed = []
            for (i, fn) in enumerate(feeded_vars_filelist):
                batch_feed.append(reader(batch_size, fn, feeded_vars_dims[i]))
            return batch_feed
        prog = self._load_program(os.path.join(config.dump_model_dir, config.dump_program_filename), config.is_text_dump_program)
        if config.is_text_dump_program:
            model_filename = self._program_type_trans(config.dump_model_dir, config.dump_program_filename, config.is_text_dump_program)
        saved_params = [v for v in prog.list_vars() if paddle.static.io.is_persistable(v)]
        print(f'persistable vars in dump program: {[v.name for v in saved_params]}')

        def check_not_expected_ops(prog, not_expected_op_types):
            if False:
                return 10
            op_types_set = set()
            for op in prog.global_block().ops:
                if op.type in not_expected_op_types and op.type not in op_types_set:
                    op_types_set.add(op.type)
            return op_types_set
        not_expected_op_types = check_not_expected_ops(prog, ['lookup_table'])
        if len(not_expected_op_types) > 0:
            print("find op type '{}' in program, please check if your program is pruned correctly !".format(list(not_expected_op_types)))
            return False
        place = framework.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            (inference_program, feed_target_names, fetch_targets) = paddle.distributed.io.load_inference_model_distributed(config.dump_model_dir, exe, model_filename=model_filename, params_filename=config.save_params_filename)
            orig_para_shape = {each_var.name: tuple(each_var.desc.shape()) for each_var in saved_params}
            for each_var in saved_params:
                var_temp = paddle.static.global_scope().find_var(each_var.name)
                assert var_temp is not None, "can't not find var: " + each_var.name
                new_shape = np.array(var_temp.get_tensor()).shape
                assert each_var.name in orig_para_shape, each_var.name + 'MUST in var list'
                orig_shape = orig_para_shape.get(each_var.name)
                if new_shape != orig_shape:
                    raise RuntimeError('Shape not matching: the Program requires a parameter with a shape of ({}), while the loaded parameter (namely [ {} ]) has a shape of  ({}).'.format(orig_shape, each_var.name, new_shape))
            feed_config = config.feed_config
            fetch_config = config.fetch_config
            fetch_targets_names = [v.name for v in fetch_targets]
            if not feed_target_names:
                print('warning! no feed targets in program.')
            if not fetch_targets_names:
                print('warning! no fetch targets in program.')
            fetch_list = fetch_targets
            feed_name_list = feed_target_names
            if feed_config.feeded_vars_names is not None and feed_target_names != feed_config.feeded_vars_names:
                print('warning! feed vars in program and config are diff: feed in program: {}. feed in config {}.'.format(feed_target_names, feed_config.feeded_vars_names))
                feed_name_list = feed_config.feeded_vars_names
                global_block = inference_program.global_block()
                need_to_remove_op_index = []
                for (i, op) in enumerate(global_block.ops):
                    op.desc.set_is_target(False)
                    if op.type == 'feed':
                        need_to_remove_op_index.append(i)
                for index in need_to_remove_op_index[::-1]:
                    global_block._remove_op(index)
            if fetch_config.fetch_vars_names is not None and fetch_targets_names != fetch_config.fetch_vars_names:
                print('warning! fetch vars in program and config are diff: fetch in program: {}. fetch in config {}.'.format(fetch_targets_names, fetch_config.fetch_vars_names))
                fetch_list = [inference_program.global_block().var(i) for i in fetch_config.fetch_vars_names]
                global_block = inference_program.global_block()
                need_to_remove_op_index = []
                for (i, op) in enumerate(global_block.ops):
                    op.desc.set_is_target(False)
                    if op.type == 'fetch':
                        need_to_remove_op_index.append(i)
                for index in need_to_remove_op_index[::-1]:
                    global_block._remove_op(index)
            return_numpy = all((v.lod_level == 0 for v in fetch_list))
            feed_tensors = []
            assert len(feed_config.feeded_vars_names) == len(feed_config.feeded_vars_dims) == len(feed_config.feeded_vars_types)
            for i in range(len(feed_config.feeded_vars_names)):
                var = inference_program.global_block().var(feed_config.feeded_vars_names[i])
                if not isinstance(feed_config.feeded_vars_dims[i], (list, tuple)):
                    tensor_shape = (feed_config.feeded_vars_dims[i],)
                else:
                    tensor_shape = tuple(feed_config.feeded_vars_dims[i])
                feed_config.feeded_vars_dims[i] = tensor_shape
                var_shape = var.shape[1:]
                if tensor_shape != var_shape:
                    raise RuntimeError("feed variable '{}' shape not match. infer program  shape: {}. feed tensor shape: {}".format(feed_config.feeded_vars_names[i], var_shape, tensor_shape))
            if not feed_config.feeded_vars_filelist:
                print('generate random feed vars.')
                for i in range(len(feed_config.feeded_vars_names)):
                    var = inference_program.global_block().var(feed_config.feeded_vars_names[i])
                    if var.lod_level == 0:
                        feed_tensors.append(np.array(np.random.random(tuple([config.batch_size] + list(feed_config.feeded_vars_dims[i]))), dtype=feed_config.feeded_vars_types[i]))
                    elif var.lod_level == 1:
                        t = np.array(np.random.random(tuple([config.batch_size] + list(feed_config.feeded_vars_dims[i]))), dtype=feed_config.feeded_vars_types[i])
                        feed_tensors.append(paddle.base.create_lod_tensor(t, [[1] * config.batch_size], place))
                    else:
                        raise RuntimeError('vars with lod_level >= 2 is not supported now in this infer program check tool.')
                results = exe.run(inference_program, feed={name: feed_tensors[i] for (i, name) in enumerate(feed_name_list)}, fetch_list=fetch_list, return_numpy=return_numpy)
            else:
                print(f'load feed vars from files: {feed_config.feeded_vars_filelist}.')
                feed_vars = [inference_program.global_block().var(feed_config.feeded_vars_names[i]) for i in range(len(feed_config.feeded_vars_names))]
                feeder = paddle.base.DataFeeder(feed_list=feed_vars, place=place)
                batch_feed = feed_gen(config.batch_size, feed_config.feeded_vars_dims, feed_config.feeded_vars_filelist)
                slots = [batch_feed]
                results = exe.run(inference_program, feed=feeder.feed(slots), fetch_list=fetch_list, return_numpy=return_numpy)
            for (i, v) in enumerate(fetch_list):
                print('fetch_targets name: %s' % v.name)
                print(f'fetch_targets: {results[i]}')
            return results

def draw_block_graphviz(block, highlights=None, path='./temp.dot'):
    if False:
        while True:
            i = 10
    '\n    Generate a debug graph for block.\n    Args:\n        block(Block): a block.\n    '
    graph = GraphPreviewGenerator('some graph')
    protostr = block.desc.serialize_to_string()
    desc = framework_pb2.BlockDesc.FromString(bytes(protostr))

    def need_highlight(name):
        if False:
            for i in range(10):
                print('nop')
        if highlights is None:
            return False
        for pattern in highlights:
            assert type(pattern) is str
            if re.match(pattern, name):
                return True
        return False
    vars = {}
    for var in desc.vars:
        if var.persistable:
            varn = graph.add_param(var.name, str(var.type).replace('\n', '<br />', 1), highlight=need_highlight(var.name))
        else:
            varn = graph.add_arg(var.name, highlight=need_highlight(var.name))
        vars[var.name] = varn

    def add_op_link_var(op, var, op2var=False):
        if False:
            return 10
        for arg in var.arguments:
            if arg not in vars:
                vars[arg] = graph.add_arg(arg, highlight=need_highlight(arg))
            varn = vars[arg]
            highlight = need_highlight(op.description) or need_highlight(varn.description)
            if op2var:
                graph.add_edge(op, varn, highlight=highlight)
            else:
                graph.add_edge(varn, op, highlight=highlight)
    for op in desc.ops:
        opn = graph.add_op(op.type, highlight=need_highlight(op.type))
        for var in op.inputs:
            add_op_link_var(opn, var, False)
        for var in op.outputs:
            add_op_link_var(opn, var, True)
    graph(path, show=False)