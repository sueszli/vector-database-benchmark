"""Fleet Utils."""
import collections
import json
import logging
import math
import os
import sys
import time
import numpy as np
import paddle
from paddle import base
from paddle.base.log_helper import get_logger
from paddle.distributed.fleet.utils.fs import HDFSClient
from . import utils
__all__ = ['FleetUtil', 'GPUPSUtil']
_logger = get_logger(__name__, logging.INFO, fmt='%(asctime)s %(levelname)s: %(message)s')
fleet = None

class FleetUtil:
    """
    FleetUtil provides some common functions for users' convenience.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil
            >>> fleet_util = FleetUtil()
            >>> fleet_util.rank0_print("my log")

    """

    def __init__(self, mode='pslib'):
        if False:
            i = 10
            return i + 15
        global fleet
        if mode == 'pslib':
            from paddle.incubate.distributed.fleet.parameter_server.pslib import fleet as fleet_pslib
            fleet = fleet_pslib
        elif mode == 'transpiler':
            from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler import fleet as fleet_transpiler
            fleet = fleet_transpiler
        else:
            raise ValueError('Please choose one mode from ["pslib", "transpiler"]')

    def rank0_print(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Worker of rank 0 print some log.\n\n        Args:\n            s(str): string to print\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.rank0_print("my log")\n\n        '
        if fleet.worker_index() != 0:
            return
        print(s)
        sys.stdout.flush()

    def rank0_info(self, s):
        if False:
            return 10
        '\n        Worker of rank 0 print some log info.\n\n        Args:\n            s(str): string to log\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.rank0_info("my log info")\n\n        '
        if fleet.worker_index() != 0:
            return
        _logger.info(s)

    def rank0_error(self, s):
        if False:
            while True:
                i = 10
        '\n        Worker of rank 0 print some log error.\n\n        Args:\n            s(str): string to log\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.rank0_error("my log error")\n\n        '
        if fleet.worker_index() != 0:
            return
        _logger.error(s)

    def set_zero(self, var_name, scope=base.global_scope(), place=base.CPUPlace(), param_type='int64'):
        if False:
            i = 10
            return i + 15
        "\n        Set tensor of a Variable to zero.\n\n        Args:\n            var_name(str): name of Variable\n            scope(Scope): Scope object, default is base.global_scope()\n            place(Place): Place object, default is base.CPUPlace()\n            param_type(str): param data type, default is int64\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP('dependency on custom variables')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.set_zero(myvar.name, myscope)\n\n        "
        param = scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype(param_type)
        param.set(param_array, place)

    def print_global_auc(self, scope=base.global_scope(), stat_pos='_generated_var_2', stat_neg='_generated_var_3', print_prefix=''):
        if False:
            return 10
        '\n        Print global auc of all distributed workers.\n\n        Args:\n            scope(Scope): Scope object, default is base.global_scope()\n            stat_pos(str): name of auc pos bucket Variable\n            stat_neg(str): name of auc neg bucket Variable\n            print_prefix(str): prefix of print auc\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP(\'dependency on custom variables\')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.print_global_auc(myscope, stat_pos=stat_pos.name,\n                ...                           stat_neg=stat_neg.name)\n\n                >>> # below is part of model\n                >>> emb = my_slot_net(slots, label) # emb can be fc layer of size 1\n                >>> similarity_norm = paddle.nn.functional.sigmoid(paddle.clip(\n                ...     emb, min=-15.0, max=15.0), name="similarity_norm")\n                >>> binary_predict = paddle.concat(input=[\n                ...     paddle.subtract(\n                ...         paddle.ceil(similarity_norm),\n                ...         similarity_norm),\n                ...     similarity_norm],\n                ...     axis=1)\n                >>> auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos,\n                ...     stat_neg] = paddle.static.auc(input=binary_predict,\n                ...                                   label=label,curve=\'ROC\',\n                ...                                   num_thresholds=4096)\n\n        '
        auc_value = self.get_global_auc(scope, stat_pos, stat_neg)
        self.rank0_print(print_prefix + ' global auc = %s' % auc_value)

    def get_global_auc(self, scope=base.global_scope(), stat_pos='_generated_var_2', stat_neg='_generated_var_3'):
        if False:
            return 10
        "\n        Get global auc of all distributed workers.\n\n        Args:\n            scope(Scope): Scope object, default is base.global_scope()\n            stat_pos(str): name of auc pos bucket Variable\n            stat_neg(str): name of auc neg bucket Variable\n\n        Returns:\n            auc_value(float), total_ins_num(int)\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP('dependency on custom variables')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> auc_value, _ = fleet_util.get_global_auc(myscope,\n                ...                                          stat_pos=stat_pos,\n                ...                                          stat_neg=stat_neg)\n\n        "
        if scope.find_var(stat_pos) is None or scope.find_var(stat_neg) is None:
            self.rank0_print('not found auc bucket')
            return None
        fleet._role_maker._barrier_worker()
        pos = np.array(scope.find_var(stat_pos).get_tensor())
        old_pos_shape = np.array(pos.shape)
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        fleet._role_maker._all_reduce(pos, global_pos)
        global_pos = global_pos.reshape(old_pos_shape)
        neg = np.array(scope.find_var(stat_neg).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet._role_maker._all_reduce(neg, global_neg)
        global_neg = global_neg.reshape(old_neg_shape)
        num_bucket = len(global_pos[0])
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        total_ins_num = 0
        for i in range(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[0][index]
            total_ins_num += global_pos[0][index]
            new_neg = neg + global_neg[0][index]
            total_ins_num += global_neg[0][index]
            area += (new_neg - neg) * (pos + new_pos) / 2
            pos = new_pos
            neg = new_neg
        auc_value = None
        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)
        fleet._role_maker._barrier_worker()
        return auc_value

    def load_fleet_model_one_table(self, table_id, path):
        if False:
            print('Hello World!')
        '\n        load pslib model to one table\n\n        Args:\n            table_id(int): load model to one table, default is None, which mean\n                           load all table.\n            path(str): model path\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.load_fleet_model_one_table(1, path="hdfs:/my/model/path")\n        '
        fleet.load_one_table(table_id, path)

    def load_fleet_model(self, path, mode=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        load pslib model\n\n        Args:\n            path(str): model path\n            mode(str): 0 or 1, which means load checkpoint or delta model,\n                       default is 0\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n\n                >>> fleet_util.load_fleet_model("hdfs:/my/model/path")\n\n                >>> fleet_util.load_fleet_model("hdfs:/my/model/path", mode=0)\n\n        '
        fleet.init_server(path, mode=mode)

    def save_fleet_model(self, path, mode=0):
        if False:
            i = 10
            return i + 15
        '\n        save pslib model\n\n        Args:\n            path(str): model path\n            mode(str): 0 or 1, which means save checkpoint or delta model,\n                       default is 0\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_fleet_model("hdfs:/my/model/path")\n\n        '
        fleet.save_persistables(None, path, mode=mode)

    def _get_xbox_str(self, output_path, day, model_path, xbox_base_key, data_path, hadoop_fs_name, monitor_data={}, mode='patch'):
        if False:
            print('Hello World!')
        xbox_dict = collections.OrderedDict()
        if mode == 'base':
            xbox_dict['id'] = str(xbox_base_key)
        elif mode == 'patch':
            xbox_dict['id'] = str(int(time.time()))
        else:
            print('warning: unknown mode %s, set it to patch' % mode)
            mode = 'patch'
            xbox_dict['id'] = str(int(time.time()))
        xbox_dict['key'] = str(xbox_base_key)
        if model_path.startswith('hdfs:') or model_path.startswith('afs:'):
            model_path = model_path[model_path.find(':') + 1:]
        xbox_dict['input'] = hadoop_fs_name + model_path.rstrip('/') + '/000'
        xbox_dict['record_count'] = '111111'
        xbox_dict['partition_type'] = '2'
        xbox_dict['job_name'] = 'default_job_name'
        xbox_dict['ins_tag'] = 'feasign'
        xbox_dict['ins_path'] = data_path
        job_id_with_host = os.popen('echo -n ${JOB_ID}').read().strip()
        instance_id = os.popen('echo -n ${INSTANCE_ID}').read().strip()
        start_pos = instance_id.find(job_id_with_host)
        end_pos = instance_id.find('--')
        if start_pos != -1 and end_pos != -1:
            job_id_with_host = instance_id[start_pos:end_pos]
        xbox_dict['job_id'] = job_id_with_host
        xbox_dict['monitor_data'] = ''
        xbox_dict['monitor_path'] = output_path.rstrip('/') + '/monitor/' + day + '.txt'
        xbox_dict['mpi_size'] = str(fleet.worker_num())
        return json.dumps(xbox_dict)

    def write_model_donefile(self, output_path, day, pass_id, xbox_base_key, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME', donefile_name='donefile.txt'):
        if False:
            print('Hello World!')
        '\n        write donefile when save model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n            xbox_base_key(str|int): xbox base key\n            hadoop_fs_name(str): hdfs/afs fs name\n            hadoop_fs_ugi(str): hdfs/afs fs ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            donefile_name(str): donefile name, default is "donefile.txt"\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.write_model_donefile(output_path="hdfs:/my/output",\n                ...                                 day=20190723,\n                ...                                 pass_id=66,\n                ...                                 xbox_base_key=int(time.time()),\n                ...                                 hadoop_fs_name="hdfs://xxx",\n                ...                                 hadoop_fs_ugi="user,passwd")\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)
        if pass_id != '-1':
            suffix_name = f'/{day}/{pass_id}/'
            model_path = output_path.rstrip('/') + suffix_name
        else:
            suffix_name = '/%s/0/' % day
            model_path = output_path.rstrip('/') + suffix_name
        if fleet.worker_index() == 0:
            donefile_path = output_path + '/' + donefile_name
            content = '%s\t%lu\t%s\t%s\t%d' % (day, xbox_base_key, model_path, pass_id, 0)
            configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                pre_content_list = pre_content.split('\n')
                day_list = [i.split('\t')[0] for i in pre_content_list]
                pass_list = [i.split('\t')[3] for i in pre_content_list]
                exist = False
                for i in range(len(day_list)):
                    if int(day) == int(day_list[i]) and int(pass_id) == int(pass_list[i]):
                        exist = True
                        break
                if not exist:
                    with open(donefile_name, 'w') as f:
                        f.write(pre_content + '\n')
                        f.write(content + '\n')
                    client.delete(donefile_path)
                    client.upload(donefile_name, output_path)
                    self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')
                else:
                    self.rank0_error(f'not write {donefile_name} because {day}/{pass_id} already exists')
            else:
                with open(donefile_name, 'w') as f:
                    f.write(content + '\n')
                client.upload(donefile_name, output_path)
                self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')
        fleet._role_maker._barrier_worker()

    def write_xbox_donefile(self, output_path, day, pass_id, xbox_base_key, data_path, hadoop_fs_name, hadoop_fs_ugi, monitor_data={}, hadoop_home='$HADOOP_HOME', donefile_name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        write delta donefile or xbox base donefile\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day of model\n            pass_id(str|int): training pass id of model\n            xbox_base_key(str|int): xbox base key\n            data_path(str|list): training data path\n            hadoop_fs_name(str): hdfs/afs fs name\n            hadoop_fs_ugi(str): hdfs/afs fs ugi\n            monitor_data(dict): metrics\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            donefile_name(str): donefile name, default is None"\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.write_xbox_donefile(\n                ...     output_path="hdfs:/my/output/",\n                ...     day=20190722,\n                ...     pass_id=1,\n                ...     xbox_base_key=int(time.time()),\n                ...     data_path="hdfs:/my/data/",\n                ...     hadoop_fs_name="hdfs://xxx",\n                ...     hadoop_fs_ugi="user,passwd",\n                ...     monitor_data={})\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)
        mode = None
        if pass_id != '-1':
            mode = 'patch'
            suffix_name = f'/{day}/delta-{pass_id}/'
            model_path = output_path.rstrip('/') + suffix_name
            if donefile_name is None:
                donefile_name = 'xbox_patch_done.txt'
        else:
            mode = 'base'
            suffix_name = '/%s/base/' % day
            model_path = output_path.rstrip('/') + suffix_name
            if donefile_name is None:
                donefile_name = 'xbox_base_done.txt'
        if isinstance(data_path, list):
            data_path = ','.join(data_path)
        if fleet.worker_index() == 0:
            donefile_path = output_path + '/' + donefile_name
            xbox_str = self._get_xbox_str(output_path, day, model_path, xbox_base_key, data_path, hadoop_fs_name, monitor_data={}, mode=mode)
            configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                last_dict = json.loads(pre_content.split('\n')[-1])
                last_day = last_dict['input'].split('/')[-3]
                last_pass = last_dict['input'].split('/')[-2].split('-')[-1]
                exist = False
                if int(day) < int(last_day) or (int(day) == int(last_day) and int(pass_id) <= int(last_pass)):
                    exist = True
                if not exist:
                    with open(donefile_name, 'w') as f:
                        f.write(pre_content + '\n')
                        f.write(xbox_str + '\n')
                    client.delete(donefile_path)
                    client.upload(donefile_name, output_path)
                    self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')
                else:
                    self.rank0_error(f'not write {donefile_name} because {day}/{pass_id} already exists')
            else:
                with open(donefile_name, 'w') as f:
                    f.write(xbox_str + '\n')
                client.upload(donefile_name, output_path)
                self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')
        fleet._role_maker._barrier_worker()

    def write_cache_donefile(self, output_path, day, pass_id, key_num, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME', donefile_name='sparse_cache.meta', **kwargs):
        if False:
            return 10
        '\n        write cache donefile\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day of model\n            pass_id(str|int): training pass id of model\n            key_num(str|int): save cache return value\n            hadoop_fs_name(str): hdfs/afs fs name\n            hadoop_fs_ugi(str): hdfs/afs fs ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            donefile_name(str): donefile name, default is "sparse_cache.meta"\n            kwargs(dict): user defined properties\n                          file_num(int): cache file num\n                          table_id(int): cache table id\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.write_cache_donefile(\n                ...     output_path="hdfs:/my/output/",\n                ...     day=20190722,\n                ...     pass_id=1,\n                ...     key_num=123456,\n                ...     hadoop_fs_name="hdfs://xxx",\n                ...     hadoop_fs_ugi="user,passwd")\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        key_num = int(key_num)
        file_num = kwargs.get('file_num', 16)
        table_id = kwargs.get('table_id', 0)
        if pass_id != '-1':
            suffix_name = '/%s/delta-%s/%03d_cache' % (day, pass_id, table_id)
            model_path = output_path.rstrip('/') + suffix_name
        else:
            suffix_name = '/%s/base/%03d_cache' % (day, table_id)
            model_path = output_path.rstrip('/') + suffix_name
        if fleet.worker_index() == 0:
            donefile_path = model_path + '/' + donefile_name
            configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                self.rank0_error('not write because %s already exists' % donefile_path)
            else:
                meta_str = 'file_prefix:part\npart_num:%s\nkey_num:%d\n' % (file_num, key_num)
                with open(donefile_name, 'w') as f:
                    f.write(meta_str)
                client.upload(donefile_name, model_path)
                self.rank0_error('write %s succeed' % donefile_path)
        fleet._role_maker._barrier_worker()

    def load_model(self, output_path, day, pass_id):
        if False:
            print('Hello World!')
        '\n        load pslib model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.load_model("hdfs:/my/path", 20190722, 88)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = f'/{day}/{pass_id}/'
        load_path = output_path + suffix_name
        self.rank0_error('going to load_model %s' % load_path)
        self.load_fleet_model(load_path)
        self.rank0_error('load_model done')

    def save_model(self, output_path, day, pass_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        save pslib model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_model("hdfs:/my/path", 20190722, 88)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = f'/{day}/{pass_id}/'
        model_path = output_path + suffix_name
        self.rank0_print('going to save_model %s' % model_path)
        self.save_fleet_model(model_path)
        self.rank0_print('save_model done')

    def save_batch_model(self, output_path, day):
        if False:
            return 10
        '\n        save batch model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_batch_model("hdfs:/my/path", 20190722)\n\n        '
        day = str(day)
        suffix_name = '/%s/0/' % day
        model_path = output_path + suffix_name
        self.rank0_print('going to save_model %s' % model_path)
        fleet.save_persistables(None, model_path, mode=3)
        self.rank0_print('save_batch_model done')

    def save_delta_model(self, output_path, day, pass_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        save delta model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_delta_model("hdfs:/my/path", 20190722, 88)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = f'/{day}/delta-{pass_id}/'
        model_path = output_path + suffix_name
        self.rank0_print('going to save_delta_model %s' % model_path)
        fleet.save_persistables(None, model_path, mode=1)
        self.rank0_print('save_delta_model done')

    def save_xbox_base_model(self, output_path, day):
        if False:
            i = 10
            return i + 15
        '\n        save xbox base model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_xbox_base_model("hdfs:/my/path", 20190722)\n\n        '
        day = str(day)
        suffix_name = '/%s/base/' % day
        model_path = output_path + suffix_name
        self.rank0_print('going to save_xbox_base_model ' + model_path)
        fleet.save_persistables(None, model_path, mode=2)
        self.rank0_print('save_xbox_base_model done')

    def save_cache_model(self, output_path, day, pass_id, mode=1, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        save cache model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n            mode(str|int): save mode\n            kwargs(dict): user defined properties\n                          table_id(int): table id to save cache\n\n        Returns:\n            key_num(int): cache key num\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_cache_model("hdfs:/my/path", 20190722, 88)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        mode = int(mode)
        table_id = kwargs.get('table_id', 0)
        suffix_name = f'/{day}/delta-{pass_id}'
        model_path = output_path.rstrip('/') + suffix_name
        self.rank0_print('going to save_cache_model %s' % model_path)
        key_num = fleet.save_cache_model(None, model_path, mode=mode, table_id=table_id)
        self.rank0_print('save_cache_model done')
        return key_num

    def save_cache_base_model(self, output_path, day, **kwargs):
        if False:
            while True:
                i = 10
        '\n        save cache model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n            kwargs(dict): user defined properties\n                          table_id(int): table id to save cache\n\n        Returns:\n            key_num(int): cache key num\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_cache_base_model("hdfs:/my/path", 20190722)\n\n        '
        day = str(day)
        table_id = kwargs.get('table_id', 0)
        suffix_name = '/%s/base' % day
        model_path = output_path.rstrip('/') + suffix_name
        self.rank0_print('going to save_cache_base_model %s' % model_path)
        key_num = fleet.save_cache_model(None, model_path, mode=2, table_id=table_id)
        self.rank0_print('save_cache_base_model done')
        return key_num

    def pull_all_dense_params(self, scope, program):
        if False:
            for i in range(10):
                print('nop')
        "\n        pull all dense params in trainer of rank 0\n\n        Args:\n            scope(Scope): base Scope\n            program(Program): base Program\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP('dependency on custom variables')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.pull_all_dense_params(my_scope, my_program)\n\n        "
        fleet._role_maker._barrier_worker()
        if fleet._role_maker.is_first_worker():
            prog_id = str(id(program))
            tables = fleet._opt_info['program_id_to_worker'][prog_id].get_desc().dense_table
            prog_conf = fleet._opt_info['program_configs'][prog_id]
            prog_tables = {}
            for key in prog_conf:
                if 'dense' not in key:
                    continue
                for table_id in prog_conf[key]:
                    prog_tables[int(table_id)] = 0
            for table in tables:
                if int(table.table_id) not in prog_tables:
                    continue
                var_name_list = []
                for i in range(0, len(table.dense_variable_name)):
                    var_name = table.dense_variable_name[i]
                    if scope.find_var(var_name) is None:
                        raise ValueError('var ' + var_name + ' not found in scope ' + 'when pull dense')
                    var_name_list.append(var_name)
                fleet._fleet_ptr.pull_dense(scope, int(table.table_id), var_name_list)
        fleet._role_maker._barrier_worker()

    def save_paddle_inference_model(self, executor, scope, program, feeded_vars, target_vars, output_path, day, pass_id, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME', save_combine=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        save paddle inference model, and upload to hdfs dnn_plugin path\n\n        Args:\n            executor(Executor): base Executor\n            scope(Scope): base Scope\n            program(Program): base Program\n            feeded_vars(list[Variable]): feed vars\n            target_vars(list[variable]): fetch vars\n            output_path(str): hdfs/afs output path\n            day(str|int): training day\n            pass_id(str|int): training pass\n            hadoop_fs_name(str): hadoop fs name\n            hadoop_fs_ugi(str): hadoop fs ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            save_combine(bool): whether to save in a file or separate files,\n                                default is True\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP(\'dependency on custom variables\')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_paddle_inference_model(exe,\n                ...                                        join_scope,\n                ...                                        join_program,\n                ...                                        feeded_vars,\n                ...                                        target_vars,\n                ...                                        "hdfs:/my/output/path/",\n                ...                                        day=20190727,\n                ...                                        pass_id=6,\n                ...                                        hadoop_fs_name="xxx",\n                ...                                        hadoop_fs_ugi="xxx,xxx")\n        '
        day = str(day)
        pass_id = str(pass_id)
        model_name = 'inference_model'
        self.pull_all_dense_params(scope, program)
        if fleet.worker_index() == 0:
            with base.scope_guard(scope):
                if save_combine:
                    paddle.static.io.save_inference_model(model_name, feeded_vars, target_vars, executor, program=program.clone())
                else:
                    paddle.static.io.save_inference_model(model_name, feeded_vars, target_vars, executor, program=program.clone())
            configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
            client = HDFSClient(hadoop_home, configs)
            if pass_id == '-1':
                dest = f'{output_path}/{day}/base/dnn_plugin/'
            else:
                dest = f'{output_path}/{day}/delta-{pass_id}/dnn_plugin/'
            if not client.is_exist(dest):
                client.makedirs(dest)
            client.upload(model_name, dest, multi_processes=5, overwrite=True)
        fleet._role_maker._barrier_worker()

    def save_paddle_params(self, executor, scope, program, model_name, output_path, day, pass_id, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME', var_names=None, save_combine=True):
        if False:
            print('Hello World!')
        '\n        save paddle model, and upload to hdfs dnn_plugin path\n\n        Args:\n            executor(Executor): base Executor\n            scope(Scope): base Scope\n            program(Program): base Program\n            model_name(str): save model local dir or filename\n            output_path(str): hdfs/afs output path\n            day(str|int): training day\n            pass_id(str|int): training pass\n            hadoop_fs_name(str): hadoop fs name\n            hadoop_fs_ugi(str): hadoop fs ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            var_names(list): save persistable var names, default is None\n            save_combine(bool): whether to save in a file or separate files,\n                                default is True\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP(\'dependency on custom variables\')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.save_paddle_params(exe,\n                ...                               join_scope,\n                ...                               join_program,\n                ...                               "paddle_dense.model.0",\n                ...                               "hdfs:/my/output/path/",\n                ...                               day=20190727,\n                ...                               pass_id=6,\n                ...                               hadoop_fs_name="xxx",\n                ...                               hadoop_fs_ugi="xxx,xxx",\n                ...                               var_names=join_all_var_names)\n                >>> fleet_util.save_paddle_params(exe,\n                ...                               join_scope,\n                ...                               join_program,\n                ...                               "paddle_dense.model.usr.0",\n                ...                               "hdfs:/my/output/path/",\n                ...                               day=20190727,\n                ...                               pass_id=6,\n                ...                               hadoop_fs_name="xxx",\n                ...                               hadoop_fs_ugi="xxx,xxx",\n                ...                               var_names=join_user_var_names)\n                >>> fleet_util.save_paddle_params(exe,\n                ...                               join_scope,\n                ...                               join_program,\n                ...                               "paddle_dense.model.item.0",\n                ...                               "hdfs:/my/output/path/",\n                ...                               day=20190727,\n                ...                               pass_id=6,\n                ...                               hadoop_fs_name="xxx",\n                ...                               hadoop_fs_ugi="xxx,xxx",\n                ...                               var_names=join_user_item_names)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        self.pull_all_dense_params(scope, program)
        if fleet.worker_index() == 0:
            vars = [program.global_block().var(i) for i in var_names]
            with base.scope_guard(scope):
                if save_combine:
                    paddle.static.io.save_vars(executor, './', program, vars=vars, filename=model_name)
                else:
                    paddle.static.io.save_vars(executor, model_name, program, vars=vars)
            configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
            client = HDFSClient(hadoop_home, configs)
            if pass_id == '-1':
                dest = f'{output_path}/{day}/base/dnn_plugin/'
            else:
                dest = f'{output_path}/{day}/delta-{pass_id}/dnn_plugin/'
            if not client.is_exist(dest):
                client.mkdirs(dest)
            client.upload(model_name, dest, multi_processes=5, overwrite=True)
        fleet._role_maker._barrier_worker()

    def get_last_save_xbox_base(self, output_path, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME'):
        if False:
            i = 10
            return i + 15
        '\n        get last saved base xbox info from xbox_base_done.txt\n\n        Args:\n            output_path(str): output path\n            hadoop_fs_name(str): hdfs/afs fs_name\n            hadoop_fs_ugi(str): hdfs/afs fs_ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n\n        Returns:\n            [last_save_day, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> last_save_day, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_xbox_base("hdfs:/my/path",\n                ...                                        hadoop_fs_name="hdfs://xxx",\n                ...                                        hadoop_fs_ugi="user,passwd")\n\n        '
        donefile_path = output_path + '/xbox_base_done.txt'
        configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, int(time.time())]
        pre_content = client.cat(donefile_path)
        last_dict = json.loads(pre_content.split('\n')[-1])
        last_day = int(last_dict['input'].split('/')[-3])
        last_path = '/'.join(last_dict['input'].split('/')[:-1])
        xbox_base_key = int(last_dict['key'])
        return [last_day, last_path, xbox_base_key]

    def get_last_save_xbox(self, output_path, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME'):
        if False:
            i = 10
            return i + 15
        '\n        get last saved xbox info from xbox_patch_done.txt\n\n        Args:\n            output_path(str): output path\n            hadoop_fs_name(str): hdfs/afs fs_name\n            hadoop_fs_ugi(str): hdfs/afs fs_ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n\n        Returns:\n            [last_save_day, last_save_pass, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_save_pass(int): pass id of saved\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> last_save_day, last_save_pass, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_xbox("hdfs:/my/path",\n                ...                                   hadoop_fs_name="hdfs://xxx",\n                ...                                   hadoop_fs_ugi="user,passwd")\n\n        '
        donefile_path = output_path + '/xbox_patch_done.txt'
        configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, '', int(time.time())]
        pre_content = client.cat(donefile_path)
        last_dict = json.loads(pre_content.split('\n')[-1])
        last_day = int(last_dict['input'].split('/')[-3])
        last_pass = int(last_dict['input'].split('/')[-2].split('-')[-1])
        last_path = '/'.join(last_dict['input'].split('/')[:-1])
        xbox_base_key = int(last_dict['key'])
        return [last_day, last_pass, last_path, xbox_base_key]

    def get_last_save_model(self, output_path, hadoop_fs_name, hadoop_fs_ugi, hadoop_home='$HADOOP_HOME'):
        if False:
            return 10
        '\n        get last saved model info from donefile.txt\n\n        Args:\n            output_path(str): output path\n            hadoop_fs_name(str): hdfs/afs fs_name\n            hadoop_fs_ugi(str): hdfs/afs fs_ugi\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n\n        Returns:\n            [last_save_day, last_save_pass, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_save_pass(int): pass id of saved\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> last_save_day, last_save_pass, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_model("hdfs:/my/path",\n                ...                                    hadoop_fs_name="hdfs://xxx",\n                ...                                    hadoop_fs_ugi="user,passwd")\n\n        '
        last_save_day = -1
        last_save_pass = -1
        last_path = ''
        donefile_path = output_path + '/donefile.txt'
        configs = {'fs.default.name': hadoop_fs_name, 'hadoop.job.ugi': hadoop_fs_ugi}
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, '', int(time.time())]
        content = client.cat(donefile_path)
        content = content.split('\n')[-1].split('\t')
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        return [last_save_day, last_save_pass, last_path, xbox_base_key]

    def get_online_pass_interval(self, days, hours, split_interval, split_per_pass, is_data_hourly_placed):
        if False:
            while True:
                i = 10
        '\n        get online pass interval\n\n        Args:\n            days(str): days to train\n            hours(str): hours to train\n            split_interval(int|str): split interval\n            split_per_pass(int}str): split per pass\n            is_data_hourly_placed(bool): is data hourly placed\n\n        Returns:\n            online_pass_interval(list)\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> online_pass_interval = fleet_util.get_online_pass_interval(\n                ...     days="{20190720..20190729}",\n                ...     hours="{0..23}",\n                ...     split_interval=5,\n                ...     split_per_pass=2,\n                ...     is_data_hourly_placed=False)\n\n        '
        days = os.popen('echo -n ' + days).read().split(' ')
        hours = os.popen('echo -n ' + hours).read().split(' ')
        split_interval = int(split_interval)
        split_per_pass = int(split_per_pass)
        splits_per_day = (int(hours[-1]) - int(hours[0]) + 1) * 60 // split_interval
        pass_per_day = splits_per_day // split_per_pass
        left_train_hour = int(hours[0])
        right_train_hour = int(hours[-1])
        start = 0
        split_path = []
        for i in range(splits_per_day):
            h = start // 60
            m = start % 60
            if h < left_train_hour or h > right_train_hour:
                start += split_interval
                continue
            if is_data_hourly_placed:
                split_path.append('%02d' % h)
            else:
                split_path.append('%02d%02d' % (h, m))
            start += split_interval
        start = 0
        online_pass_interval = []
        for i in range(pass_per_day):
            online_pass_interval.append([])
            for j in range(start, start + split_per_pass):
                online_pass_interval[i].append(split_path[j])
            start += split_per_pass
        return online_pass_interval

    def get_global_metrics(self, scope=base.global_scope(), stat_pos_name='_generated_var_2', stat_neg_name='_generated_var_3', sqrerr_name='sqrerr', abserr_name='abserr', prob_name='prob', q_name='q', pos_ins_num_name='pos', total_ins_num_name='total'):
        if False:
            print('Hello World!')
        '\n        get global metrics, including auc, bucket_error, mae, rmse,\n        actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num.\n\n        Args:\n            scope(Scope): Scope object, default is base.global_scope()\n            stat_pos_name(str): name of auc pos bucket Variable\n            stat_neg_name(str): name of auc neg bucket Variable\n            sqrerr_name(str): name of sqrerr Variable\n            abserr_name(str): name of abserr Variable\n            prob_name(str): name of prob Variable\n            q_name(str): name of q Variable\n            pos_ins_num_name(str): name of pos ins num Variable\n            total_ins_num_name(str): name of total ins num Variable\n\n        Returns:\n            [auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc,\n             mean_predict_qvalue, total_ins_num]\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP(\'dependency on custom variables\')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> metric_list = fleet_util.get_global_metrics(myscope,\n                ...                                             stat_pos.name,\n                ...                                             stat_neg.name,\n                ...                                             local_sqrerr.name,\n                ...                                             local_abserr.name,\n                ...                                             local_prob.name,\n                ...                                             local_q.name,\n                ...                                             local_pos_ins.name,\n                ...                                             local_total_ins.name)\n\n                >>> # below is part of example model\n                >>> label = paddle.static.data(name="click", shape=[-1, 1],\\\n                ...     dtype="int64", lod_level=0)\n                >>> emb = my_slot_net(slots, label) # emb can be fc layer of size 1\n                >>> similarity_norm = paddle.nn.functional.sigmoid(paddle.clip(\\\n                ...     emb, min=-15.0, max=15.0), name="similarity_norm")\\\n                >>> binary_predict = paddle.concat(input=[\\\n                ...     paddle.subtract(\\\n                ...         paddle.ceil(similarity_norm), similarity_norm),\\\n                ...     similarity_norm], axis=1)\n                >>> auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \\\n                ...     stat_neg] = paddle.static.auc(input=binary_predict,\\\n                ...                                  label=label, curve=\'ROC\',\\\n                ...                                  num_thresholds=4096)\n                >>> local_sqrerr, local_abserr, local_prob, local_q, local_pos_ins,\\\n                ...     local_total_ins = paddle.static.ctr_metric_bundle(\\\n                ...         similarity_norm, label)\n\n        '
        if scope.find_var(stat_pos_name) is None or scope.find_var(stat_neg_name) is None:
            self.rank0_print('not found auc bucket')
            return [None] * 9
        elif scope.find_var(sqrerr_name) is None:
            self.rank0_print('not found sqrerr_name=%s' % sqrerr_name)
            return [None] * 9
        elif scope.find_var(abserr_name) is None:
            self.rank0_print('not found abserr_name=%s' % abserr_name)
            return [None] * 9
        elif scope.find_var(prob_name) is None:
            self.rank0_print('not found prob_name=%s' % prob_name)
            return [None] * 9
        elif scope.find_var(q_name) is None:
            self.rank0_print('not found q_name=%s' % q_name)
            return [None] * 9
        elif scope.find_var(pos_ins_num_name) is None:
            self.rank0_print('not found pos_ins_num_name=%s' % pos_ins_num_name)
            return [None] * 9
        elif scope.find_var(total_ins_num_name) is None:
            self.rank0_print('not found total_ins_num_name=%s' % total_ins_num_name)
            return [None] * 9
        fleet._role_maker._barrier_worker()
        auc = self.get_global_auc(scope, stat_pos_name, stat_neg_name)
        pos = np.array(scope.find_var(stat_pos_name).get_tensor())
        old_pos_shape = np.array(pos.shape)
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        fleet._role_maker._all_reduce(pos, global_pos)
        global_pos = global_pos.reshape(old_pos_shape)
        neg = np.array(scope.find_var(stat_neg_name).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet._role_maker._all_reduce(neg, global_neg)
        global_neg = global_neg.reshape(old_neg_shape)
        num_bucket = len(global_pos[0])

        def get_metric(name):
            if False:
                for i in range(10):
                    print('nop')
            metric = np.array(scope.find_var(name).get_tensor())
            old_metric_shape = np.array(metric.shape)
            metric = metric.reshape(-1)
            global_metric = np.copy(metric) * 0
            fleet._role_maker._all_reduce(metric, global_metric)
            global_metric = global_metric.reshape(old_metric_shape)
            return global_metric[0]
        global_sqrerr = get_metric(sqrerr_name)
        global_abserr = get_metric(abserr_name)
        global_prob = get_metric(prob_name)
        global_q_value = get_metric(q_name)
        pos_ins_num = get_metric(pos_ins_num_name)
        total_ins_num = get_metric(total_ins_num_name)
        neg_ins_num = total_ins_num - pos_ins_num
        mae = global_abserr / total_ins_num
        rmse = math.sqrt(global_sqrerr / total_ins_num)
        return_actual_ctr = pos_ins_num / total_ins_num
        predicted_ctr = global_prob / total_ins_num
        mean_predict_qvalue = global_q_value / total_ins_num
        copc = 0.0
        if abs(predicted_ctr > 1e-06):
            copc = return_actual_ctr / predicted_ctr
        last_ctr = -1.0
        impression_sum = 0.0
        ctr_sum = 0.0
        click_sum = 0.0
        error_sum = 0.0
        error_count = 0.0
        click = 0.0
        show = 0.0
        ctr = 0.0
        adjust_ctr = 0.0
        relative_error = 0.0
        actual_ctr = 0.0
        relative_ctr_error = 0.0
        k_max_span = 0.01
        k_relative_error_bound = 0.05
        for i in range(num_bucket):
            click = global_pos[0][i]
            show = global_pos[0][i] + global_neg[0][i]
            ctr = float(i) / num_bucket
            if abs(ctr - last_ctr) > k_max_span:
                last_ctr = ctr
                impression_sum = 0.0
                ctr_sum = 0.0
                click_sum = 0.0
            impression_sum += show
            ctr_sum += ctr * show
            click_sum += click
            if impression_sum == 0:
                continue
            adjust_ctr = ctr_sum / impression_sum
            if adjust_ctr == 0:
                continue
            relative_error = math.sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum))
            if relative_error < k_relative_error_bound:
                actual_ctr = click_sum / impression_sum
                relative_ctr_error = abs(actual_ctr / adjust_ctr - 1)
                error_sum += relative_ctr_error * impression_sum
                error_count += impression_sum
                last_ctr = -1
        bucket_error = error_sum / error_count if error_count > 0 else 0.0
        return [auc, bucket_error, mae, rmse, return_actual_ctr, predicted_ctr, copc, mean_predict_qvalue, int(total_ins_num)]

    def print_global_metrics(self, scope=base.global_scope(), stat_pos_name='_generated_var_2', stat_neg_name='_generated_var_3', sqrerr_name='sqrerr', abserr_name='abserr', prob_name='prob', q_name='q', pos_ins_num_name='pos', total_ins_num_name='total', print_prefix=''):
        if False:
            print('Hello World!')
        '\n        print global metrics, including auc, bucket_error, mae, rmse,\n        actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num.\n\n        Args:\n            scope(Scope): Scope object, default is base.global_scope()\n            stat_pos_name(str): name of auc pos bucket Variable\n            stat_neg_name(str): name of auc neg bucket Variable\n            sqrerr_name(str): name of sqrerr Variable\n            abserr_name(str): name of abserr Variable\n            prob_name(str): name of prob Variable\n            q_name(str): name of q Variable\n            pos_ins_num_name(str): name of pos ins num Variable\n            total_ins_num_name(str): name of total ins num Variable\n            print_prefix(str): print prefix\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> # doctest: +SKIP(\'dependency on custom variables\')\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> fleet_util.print_global_metrics(myscope,\n                ...                                 stat_pos.name,\n                ...                                 stat_neg.name,\n                ...                                 local_sqrerr.name,\n                ...                                 local_abserr.name,\n                ...                                 local_prob.name,\n                ...                                 local_q.name,\n                ...                                 local_pos_ins.name,\n                ...                                 local_total_ins.name)\n\n                >>> # below is part of model\n                >>> label = paddle.static.data(name="click", shape=[-1, 1],\\\n                ...     dtype="int64", lod_level=0)\n                >>> emb = my_slot_net(slots, label) # emb can be fc layer of size 1\n                >>> similarity_norm = paddle.nn.functional.sigmoid(paddle.clip(\\\n                ...     emb, min=-15.0, max=15.0), name="similarity_norm")\\\n                >>> binary_predict = paddle.concat(input=[\\\n                ...     paddle.subtract(\\\n                ...         paddle.ceil(similarity_norm), similarity_norm),\\\n                ...     similarity_norm], axis=1)\n                >>> auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \\\n                ...     stat_neg] = paddle.static.auc(input=binary_predict,\\\n                ...                                  label=label, curve=\'ROC\',\\\n                ...                                  num_thresholds=4096)\n                >>> local_sqrerr, local_abserr, local_prob, local_q, local_pos_ins, \\\n                ...     local_total_ins = paddle.static.ctr_metric_bundle(\\\n                ...         similarity_norm, label)\n\n        '
        if scope.find_var(stat_pos_name) is None or scope.find_var(stat_neg_name) is None:
            self.rank0_print('not found auc bucket')
            return
        elif scope.find_var(sqrerr_name) is None:
            self.rank0_print('not found sqrerr_name=%s' % sqrerr_name)
            return
        elif scope.find_var(abserr_name) is None:
            self.rank0_print('not found abserr_name=%s' % abserr_name)
            return
        elif scope.find_var(prob_name) is None:
            self.rank0_print('not found prob_name=%s' % prob_name)
            return
        elif scope.find_var(q_name) is None:
            self.rank0_print('not found q_name=%s' % q_name)
            return
        elif scope.find_var(pos_ins_num_name) is None:
            self.rank0_print('not found pos_ins_num_name=%s' % pos_ins_num_name)
            return
        elif scope.find_var(total_ins_num_name) is None:
            self.rank0_print('not found total_ins_num_name=%s' % total_ins_num_name)
            return
        (auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num) = self.get_global_metrics(scope, stat_pos_name, stat_neg_name, sqrerr_name, abserr_name, prob_name, q_name, pos_ins_num_name, total_ins_num_name)
        self.rank0_print('{} global AUC={:.6f} BUCKET_ERROR={:.6f} MAE={:.6f} RMSE={:.6f} Actural_CTR={:.6f} Predicted_CTR={:.6f} COPC={:.6f} MEAN Q_VALUE={:.6f} Ins number={}'.format(print_prefix, auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num))

    def program_type_trans(self, prog_dir, prog_fn, is_text):
        if False:
            while True:
                i = 10
        return utils.program_type_trans(prog_dir, prog_fn, is_text)

    def load_program(self, model_filename, is_text):
        if False:
            i = 10
            return i + 15
        return utils.load_program(model_filename, is_text)

    def draw_from_program_file(self, model_filename, is_text, output_dir, output_filename):
        if False:
            i = 10
            return i + 15
        'draw program from file'
        program = self.load_program(model_filename, is_text)
        utils.graphviz(program.global_block(), output_dir, output_filename)

    def draw_from_program(self, program, output_dir, output_name):
        if False:
            return 10
        'draw Program'
        utils.graphviz(program.global_block(), output_dir, output_name)

    def check_two_programs(self, config):
        if False:
            return 10
        train_prog = self.load_program(config.train_prog_path, config.is_text_train_program)
        pruned_prog = self.load_program(config.pruned_prog_path, config.is_text_pruned_program)
        if config.draw:
            pruned_dir = os.path.dirname(config.pruned_prog_path)
            self.draw_from_program(pruned_prog, pruned_dir, config.draw_out_name)
        res = utils.check_pruned_program_vars(train_prog, pruned_prog)
        if res:
            _logger.info('check_programs succeed.')
        else:
            _logger.info('check_programs failed. pruned program and train program not match!')
        return res

    def check_vars_and_dump(self, config):
        if False:
            while True:
                i = 10
        _logger.info('start check_vars_and_dump.')
        results = utils.check_saved_vars_try_dump(config.dump_model_dir, config.dump_program_filename, config.is_text_dump_program, config.feed_config, config.fetch_config, config.batch_size, config.save_params_filename)
        _logger.info('check_vars_and_dump succeed.')
        return results

    def parse_program_proto(self, prog_path, is_text, output_dir):
        if False:
            i = 10
            return i + 15
        '\n        Parse program.proto into a more readable format.\n        This function will generate three files:\n        output_dir/vars_all.log,\n        output_dir/vars_persistable.log,\n        output_dir/ops.log.\n\n        Args:\n            prog_path(str): proto file path to be parsed.\n            is_text(bool): proto file is human-readale format or not(binary).\n            output_dir(str): output dir.\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import FleetUtil\n                >>> fleet_util = FleetUtil()\n                >>> program_path = "./program.pbtxt"\n                >>> is_text = True\n                >>> output_dir = "/tmp/"\n                >>> fleet_util.parse_program_proto(program_path, is_text, output_dir)\n        '
        program = self.load_program(prog_path, is_text)
        utils.parse_program(program, output_dir)

class GPUPSUtil(FleetUtil):
    """
    GPUPSUtil provides some common functions for users' convenience.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil
            >>> fleet_util = GPUPSUtil()
            >>> fleet_util.rank0_print("my log")
    """

    def __init__(self, fs_client=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('pslib')
        self._afs = fs_client

    def init(self, fs_name, fs_user, fs_passwd, fs_conf):
        if False:
            for i in range(10):
                print('nop')
        '\n        init for fs config\n\n        Args:\n            fs_name(str): fs name\n            fs_user(str): fs user\n            fs_passwd(str): fs password\n            fs_conf(str): fs and afs conf path\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.init(20190722, 88, 88, "./afs.conf")\n        '
        self._afs.init(fs_name, fs_user, fs_passwd, fs_conf)

    def set_fsclient(self, fs_client):
        if False:
            return 10
        '\n        set fs_client for fs config\n\n        Args:\n            fs_client(AFSClient): fs_client object\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n        '
        self._afs = fs_client

    def get_last_save_xbox_base(self, output_path):
        if False:
            return 10
        '\n        get last saved base xbox info from xbox_base_done.txt\n\n        Args:\n            output_path(str): output path\n\n        Returns:\n            [last_save_day, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> last_save_day, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_xbox_base("hdfs:/my/path")\n\n        '
        donefile_path = output_path + '/xbox_base_done.txt'
        if not self._afs.is_file(donefile_path):
            return [-1, -1, int(time.time())]
        self._afs.download(donefile_path, './xbox_base_done.txt')
        pre_content = ''
        with open('xbox_base_done.txt', 'r') as f:
            pre_content = f.read()
        pre_content = pre_content.strip()
        last_dict = json.loads(pre_content.split('\n')[-1])
        last_day = int(last_dict['input'].split('/')[-3])
        last_path = '/'.join(last_dict['input'].split('/')[:-1])
        xbox_base_key = int(last_dict['key'])
        return [last_day, last_path, xbox_base_key]

    def get_last_save_xbox(self, output_path):
        if False:
            while True:
                i = 10
        '\n        get last saved xbox info from xbox_patch_done.txt\n\n        Args:\n            output_path(str): output path\n\n        Returns:\n            [last_save_day, last_save_pass, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_save_pass(int): pass id of saved\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> last_save_day, last_save_pass, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_xbox("hdfs:/my/path")\n\n        '
        donefile_path = output_path + '/xbox_patch_done.txt'
        if not self._afs.is_file(donefile_path):
            return [-1, -1, '', int(time.time())]
        self._afs.download(donefile_path, 'xbox_patch_done.txt')
        pre_content = ''
        with open('xbox_patch_done.txt', 'r') as f:
            pre_content = f.read()
        pre_content = pre_content.strip()
        last_dict = json.loads(pre_content.split('\n')[-1])
        last_day = int(last_dict['input'].split('/')[-3])
        last_pass = int(last_dict['input'].split('/')[-2].split('-')[-1])
        last_path = '/'.join(last_dict['input'].split('/')[:-1])
        xbox_base_key = int(last_dict['key'])
        os.remove('xbox_patch_done.txt')
        return [last_day, last_pass, last_path, xbox_base_key]

    def get_last_save_model(self, output_path):
        if False:
            while True:
                i = 10
        '\n        get last saved model info from donefile.txt\n\n        Args:\n            output_path(str): output path\n\n        Returns:\n            [last_save_day, last_save_pass, last_path, xbox_base_key]\n            last_save_day(int): day of saved model\n            last_save_pass(int): pass id of saved\n            last_path(str): model path\n            xbox_base_key(int): xbox key\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> last_save_day, last_save_pass, last_path, xbox_base_key = \\\n                ...     fleet_util.get_last_save_model("hdfs:/my/path")\n\n        '
        last_save_day = -1
        last_save_pass = -1
        last_path = ''
        donefile_path = output_path + '/donefile.txt'
        if not self._afs.is_file(donefile_path):
            return [-1, -1, '', int(time.time())]
        self._afs.download(donefile_path, './donefile.txt')
        content = ''
        with open('donefile.txt', 'r') as f:
            content = f.read()
        content = content.strip().split('\n')[-1].split('\t')
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        os.remove('donefile.txt')
        return [last_save_day, last_save_pass, last_path, xbox_base_key]

    def write_model_donefile(self, output_path, day, pass_id, xbox_base_key, donefile_name='donefile.txt'):
        if False:
            for i in range(10):
                print('nop')
        '\n        write donefile when save model\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day\n            pass_id(str|int): training pass id\n            xbox_base_key(str|int): xbox base key\n            donefile_name(str): donefile name, default is "donefile.txt"\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> fleet_util.write_model_donefile(output_path="hdfs:/my/output",\n                ...                                 day=20190723,\n                ...                                 pass_id=66,\n                ...                                 xbox_base_key=int(time.time()))\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)
        if pass_id != '-1':
            suffix_name = f'/{day}/{pass_id}/'
            model_path = output_path.rstrip('/') + suffix_name
        else:
            suffix_name = '/%s/0/' % day
            model_path = output_path.rstrip('/') + suffix_name
        if fleet.worker_index() == 0:
            donefile_path = output_path + '/' + donefile_name
            content = '%s\t%lu\t%s\t%s\t%d' % (day, xbox_base_key, model_path, pass_id, 0)
            if self._afs.is_file(donefile_path):
                self._afs.download(donefile_path, donefile_name)
                pre_content = ''
                with open(donefile_name, 'r') as f:
                    pre_content = f.read()
                pre_content_list = pre_content.strip().split('\n')
                day_list = [i.split('\t')[0] for i in pre_content_list]
                pass_list = [i.split('\t')[3] for i in pre_content_list]
                os.remove(donefile_name)
                exist = False
                for i in range(len(day_list)):
                    if int(day) == int(day_list[i]) and int(pass_id) == int(pass_list[i]):
                        exist = True
                        break
                if not exist:
                    with open(donefile_name, 'w') as f:
                        f.write(pre_content.strip() + '\n')
                        f.write(content + '\n')
                    self._afs.delete(donefile_path)
                    self._afs.upload(donefile_name, donefile_path)
                    self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')
                else:
                    self.rank0_error(f'not write {donefile_name} because {day}/{pass_id} already exists')
            else:
                with open(donefile_name, 'w') as f:
                    f.write(content + '\n')
                self._afs.upload(donefile_name, donefile_path)
                self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')

    def write_xbox_donefile(self, output_path, day, pass_id, xbox_base_key, data_path, hadoop_fs_name, hadoop_fs_ugi, monitor_data={}, hadoop_home='$HADOOP_HOME', donefile_name=None):
        if False:
            i = 10
            return i + 15
        '\n        write delta donefile or xbox base donefile\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day of model\n            pass_id(str|int): training pass id of model\n            xbox_base_key(str|int): xbox base key\n            data_path(str|list): training data path\n            monitor_data(dict): metrics\n            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"\n            donefile_name(str): donefile name, default is None"\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> fleet_util.write_xbox_donefile(\n                ...     output_path="hdfs:/my/output/",\n                ...     day=20190722,\n                ...     pass_id=1,\n                ...     xbox_base_key=int(time.time()),\n                ...     data_path="hdfs:/my/data/",\n                ...     monitor_data={})\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)
        mode = None
        if pass_id != '-1':
            mode = 'patch'
            suffix_name = f'/{day}/delta-{pass_id}/'
            model_path = output_path.rstrip('/') + suffix_name
            if donefile_name is None:
                donefile_name = 'xbox_patch_done.txt'
        else:
            mode = 'base'
            suffix_name = '/%s/base/' % day
            model_path = output_path.rstrip('/') + suffix_name
            if donefile_name is None:
                donefile_name = 'xbox_base_done.txt'
        if isinstance(data_path, list):
            data_path = ','.join(data_path)
        if fleet.worker_index() == 0:
            donefile_path = output_path + '/' + donefile_name
            xbox_str = self._get_xbox_str(output_path, day, model_path, xbox_base_key, data_path, hadoop_fs_name, monitor_data={}, mode=mode)
            if self._afs.is_exist(donefile_path):
                self.rank0_info('exist %s succeed' % donefile_path)
                self._afs.download(donefile_path, donefile_name)
                pre_content = ''
                with open(donefile_name, 'r') as f:
                    pre_content = f.read()
                last_dict = json.loads(pre_content.strip().split('\n')[-1])
                last_day = last_dict['input'].split('/')[-3]
                last_pass = last_dict['input'].split('/')[-2].split('-')[-1]
                os.remove(donefile_name)
                self.rank0_info('remove %s succeed' % donefile_name)
                exist = False
                if int(day) < int(last_day) or (int(day) == int(last_day) and int(pass_id) <= int(last_pass)):
                    exist = True
                if not exist:
                    with open(donefile_name, 'w') as f:
                        f.write(pre_content.strip() + '\n')
                        f.write(xbox_str + '\n')
                    self._afs.delete(donefile_path)
                    self._afs.upload(donefile_name, donefile_path)
                    self.rank0_info(f'write {day}/{pass_id} {donefile_name} succeed')
                else:
                    self.rank0_info(f'not write {donefile_name} because {day}/{pass_id} already exists')
            else:
                with open(donefile_name, 'w') as f:
                    f.write(xbox_str + '\n')
                self._afs.upload(donefile_name, donefile_path)
                self.rank0_error(f'write {day}/{pass_id} {donefile_name} succeed')

    def write_cache_donefile(self, output_path, day, pass_id, key_num, donefile_name='sparse_cache.meta', **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        write cache donefile\n\n        Args:\n            output_path(str): output path\n            day(str|int): training day of model\n            pass_id(str|int): training pass id of model\n            key_num(str|int): save cache return value\n            donefile_name(str): donefile name, default is "sparse_cache.meta"\n            kwargs(dict): user defined properties\n                          file_num(int): cache file num\n                          table_id(int): cache table id\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n                >>> from paddle.incubate.distributed.fleet.fleet_util import GPUPSUtil\n                >>> from paddle.distributed.fleet.utils.fs import AFSClient\n                >>> hdfs_client = AFSClient()\n                >>> fleet_util = GPUPSUtil()\n                >>> fleet_util.set_fsclient(hdfs_client)\n                >>> fleet_util.write_cache_donefile(\n                ...     output_path="hdfs:/my/output/",\n                ...     day=20190722,\n                ...     pass_id=1,\n                ...     key_num=123456)\n\n        '
        day = str(day)
        pass_id = str(pass_id)
        key_num = int(key_num)
        file_num = kwargs.get('file_num', 16)
        table_id = kwargs.get('table_id', 0)
        if pass_id != '-1':
            suffix_name = '/%s/delta-%s/%03d_cache' % (day, pass_id, table_id)
            model_path = output_path.rstrip('/') + suffix_name
        else:
            suffix_name = '/%s/base/%03d_cache' % (day, table_id)
            model_path = output_path.rstrip('/') + suffix_name
        if fleet.worker_index() == 0:
            donefile_path = model_path + '/' + donefile_name
            if self._afs.is_file(donefile_path):
                self.rank0_error('not write because %s already exists' % donefile_path)
            else:
                meta_str = 'file_prefix:part\npart_num:%s\nkey_num:%d\n' % (file_num, key_num)
                with open(donefile_name, 'w') as f:
                    f.write(meta_str)
                self._afs.upload(donefile_name, donefile_path)
                self.rank0_error('write %s succeed' % donefile_path)

    def _get_xbox_str(self, output_path, day, model_path, xbox_base_key, data_path, hadoop_fs_name, monitor_data={}, mode='patch'):
        if False:
            return 10
        xbox_dict = collections.OrderedDict()
        if mode == 'base':
            xbox_dict['id'] = str(xbox_base_key)
        elif mode == 'patch':
            xbox_dict['id'] = str(int(time.time()))
        else:
            print('warning: unknown mode %s, set it to patch' % mode)
            mode = 'patch'
            xbox_dict['id'] = str(int(time.time()))
        xbox_dict['key'] = str(xbox_base_key)
        if model_path.startswith('hdfs:') or model_path.startswith('afs:'):
            model_path = model_path[model_path.find(':') + 1:]
        xbox_dict['input'] = hadoop_fs_name + model_path.rstrip('/') + '/000'
        xbox_dict['record_count'] = '111111'
        xbox_dict['partition_type'] = '2'
        xbox_dict['job_name'] = 'default_job_name'
        xbox_dict['ins_tag'] = 'feasign'
        xbox_dict['ins_path'] = data_path
        xbox_dict['job_id'] = os.environ.get('PADDLE_JOB_ID', '')
        xbox_dict['monitor_data'] = ''
        xbox_dict['monitor_path'] = output_path.rstrip('/') + '/monitor/' + day + '.txt'
        xbox_dict['mpi_size'] = str(fleet.worker_num())
        return json.dumps(xbox_dict)