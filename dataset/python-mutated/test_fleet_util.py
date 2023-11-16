import os
import sys
import tarfile
import tempfile
import unittest
import numpy as np
import paddle
from paddle.dataset.common import download
from paddle.distributed.fleet.base import role_maker

class TestFleetUtil(unittest.TestCase):
    proto_data_url = 'https://fleet.bj.bcebos.com/fleet_util_data.tgz'
    proto_data_md5 = '59b7f12fd9dc24b64ae8e4629523a92a'
    module_name = 'fleet_util_data'
    pruned_dir = os.path.join('fleet_util_data', 'pruned_model')
    train_dir = os.path.join('fleet_util_data', 'train_program')

    def test_util_base(self):
        if False:
            while True:
                i = 10
        from paddle.distributed import fleet
        util = fleet.UtilBase()
        strategy = fleet.DistributedStrategy()
        util._set_strategy(strategy)
        role_maker = None
        util._set_role_maker(role_maker)

    def test_util_factory(self):
        if False:
            return 10
        from paddle.distributed import fleet
        factory = fleet.base.util_factory.UtilFactory()
        strategy = fleet.DistributedStrategy()
        role_maker = None
        optimize_ops = []
        params_grads = []
        context = {}
        context['role_maker'] = role_maker
        context['valid_strategy'] = strategy
        util = factory._create_util(context)
        self.assertIsNone(util.role_maker)

    def test_get_util(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle.distributed import fleet
        from paddle.distributed.fleet.base import role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertIsNotNone(fleet.util)

    def test_set_user_defined_util(self):
        if False:
            return 10
        from paddle.distributed import fleet

        class UserDefinedUtil(fleet.UtilBase):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()

            def get_user_id(self):
                if False:
                    return 10
                return 10
        from paddle.distributed.fleet.base import role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        my_util = UserDefinedUtil()
        fleet.util = my_util
        user_id = fleet.util.get_user_id()
        self.assertEqual(user_id, 10)

    def test_fs(self):
        if False:
            return 10
        from paddle.distributed import fleet
        from paddle.distributed.fleet.utils import LocalFS
        fs = LocalFS()
        (dirs, files) = fs.ls_dir('test_tmp')
        (dirs, files) = fs.ls_dir('./')
        self.assertFalse(fs.need_upload_download())
        fleet.util._set_file_system(fs)

    def download_files(self):
        if False:
            print('Hello World!')
        path = download(self.proto_data_url, self.module_name, self.proto_data_md5)
        print('data is downloaded at ' + path)
        tar = tarfile.open(path)
        unzip_folder = tempfile.mkdtemp()
        tar.extractall(unzip_folder)
        return unzip_folder

    def test_get_file_shard(self):
        if False:
            while True:
                i = 10
        from paddle.distributed import fleet
        self.assertRaises(Exception, fleet.util.get_file_shard, 'files')
        role = role_maker.UserDefinedRoleMaker(is_collective=False, init_gloo=False, current_id=0, role=role_maker.Role.WORKER, worker_endpoints=['127.0.0.1:6003', '127.0.0.1:6004'], server_endpoints=['127.0.0.1:6001', '127.0.0.1:6002'])
        fleet.init(role)
        files = fleet.util.get_file_shard(['1', '2', '3'])
        self.assertTrue(len(files) == 2 and '1' in files and ('2' in files))

    def test_program_type_trans(self):
        if False:
            return 10
        from paddle.distributed import fleet
        data_dir = self.download_files()
        program_dir = os.path.join(data_dir, self.pruned_dir)
        text_program = 'pruned_main_program.pbtxt'
        binary_program = 'pruned_main_program.bin'
        text_to_binary = fleet.util._program_type_trans(program_dir, text_program, True)
        binary_to_text = fleet.util._program_type_trans(program_dir, binary_program, False)
        self.assertTrue(os.path.exists(os.path.join(program_dir, text_to_binary)))
        self.assertTrue(os.path.exists(os.path.join(program_dir, binary_to_text)))

    def test_prams_check(self):
        if False:
            while True:
                i = 10
        from paddle.distributed import fleet
        data_dir = self.download_files()

        class config:
            pass
        feed_config = config()
        feed_config.feeded_vars_names = ['concat_1.tmp_0', 'concat_2.tmp_0']
        feed_config.feeded_vars_dims = [682, 1199]
        feed_config.feeded_vars_types = [np.float32, np.float32]
        feed_config.feeded_vars_filelist = [os.path.join(data_dir, os.path.join(self.pruned_dir, 'concat_1')), os.path.join(data_dir, os.path.join(self.pruned_dir, 'concat_2'))]
        fetch_config = config()
        fetch_config.fetch_vars_names = ['similarity_norm.tmp_0']
        conf = config()
        conf.batch_size = 1
        conf.feed_config = feed_config
        conf.fetch_config = fetch_config
        conf.dump_model_dir = os.path.join(data_dir, self.pruned_dir)
        conf.dump_program_filename = 'pruned_main_program.pbtxt'
        conf.is_text_dump_program = True
        conf.save_params_filename = None
        conf.dump_program_filename = 'pruned_main_program.save_var_shape_not_match'
        self.assertRaises(Exception, fleet.util._params_check)
        conf.dump_program_filename = 'pruned_main_program.no_feed_fetch'
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(results[0], np.array([[3.0590223e-07]], dtype=np.float32))
        conf.dump_program_filename = 'pruned_main_program.feed_var_shape_not_match'
        self.assertRaises(Exception, fleet.util._params_check)
        conf.dump_program_filename = 'pruned_main_program.pbtxt'
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)
        np.testing.assert_array_almost_equal(results[0], np.array([[3.0590223e-07]], dtype=np.float32))
        conf.feed_config.feeded_vars_filelist = None
        conf.dump_program_filename = 'pruned_main_program.feed_lod2'
        self.assertRaises(Exception, fleet.util._params_check)
        conf.dump_program_filename = 'pruned_main_program.pbtxt'
        results = fleet.util._params_check(conf)
        self.assertTrue(len(results) == 1)

    def test_proto_check(self):
        if False:
            i = 10
            return i + 15
        from paddle.distributed import fleet
        data_dir = self.download_files()

        class config:
            pass
        conf = config()
        conf.train_prog_path = os.path.join(data_dir, os.path.join(self.train_dir, 'join_main_program.pbtxt'))
        conf.is_text_train_program = True
        conf.pruned_prog_path = os.path.join(data_dir, os.path.join(self.pruned_dir, 'pruned_main_program.save_var_shape_not_match'))
        conf.is_text_pruned_program = True
        conf.draw = False
        res = fleet.util._proto_check(conf)
        self.assertFalse(res)
        conf.pruned_prog_path = os.path.join(data_dir, os.path.join(self.pruned_dir, 'pruned_main_program.pbtxt'))
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            conf.draw = False
        else:
            conf.draw = True
            conf.draw_out_name = 'pruned_check'
        res = fleet.util._proto_check(conf)
        self.assertTrue(res)

    def test_visualize(self):
        if False:
            i = 10
            return i + 15
        from paddle.distributed import fleet
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            data_dir = self.download_files()
            program_path = os.path.join(data_dir, os.path.join(self.train_dir, 'join_main_program.pbtxt'))
            is_text = True
            program = fleet.util._load_program(program_path, is_text)
            output_dir = os.path.join(data_dir, self.train_dir)
            output_filename = 'draw_prog'
            fleet.util._visualize_graphviz(program, output_dir, output_filename)
            self.assertTrue(os.path.exists(os.path.join(output_dir, output_filename + '.dot')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, output_filename + '.pdf')))

    def test_support_tuple(self):
        if False:
            return 10
        role = paddle.distributed.fleet.PaddleCloudRoleMaker(is_collective=False, init_gloo=True, path='./tmp_gloo')
        paddle.distributed.fleet.init(role)
        output_1 = paddle.distributed.fleet.util.all_reduce([3, 4], 'sum', 'all')
        output_2 = paddle.distributed.fleet.util.all_reduce((3, 4), 'sum', 'all')
        self.assertTrue(output_1 == output_2)
if __name__ == '__main__':
    unittest.main()