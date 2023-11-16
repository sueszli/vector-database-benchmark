import os
import shlex
import shutil
import sys
import tempfile
import unittest
import paddle

def get_test_file():
    if False:
        print('Hello World!')
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirname, 'distributed_fused_lamb_test_base.py')

def remove_file_if_exists(file_name):
    if False:
        while True:
            i = 10
    if not os.path.exists(file_name):
        return
    if os.path.isfile(file_name):
        os.remove(file_name)
    else:
        shutil.rmtree(file_name)

def run_test(clip_after_allreduce=True, max_global_norm=-1.0, gradient_merge_steps=1, use_master_acc_grad=True, need_env={}):
    if False:
        print('Hello World!')
    temp_dir = tempfile.TemporaryDirectory()
    if not paddle.is_compiled_with_cuda():
        return
    if os.name == 'nt':
        return
    args = locals()
    log_dir = os.path.join(temp_dir.name, f'log_{os.getpid()}')
    cmd = [sys.executable, '-u', '-m', 'paddle.distributed.launch', '--devices', '0,1', '--log_dir', log_dir, get_test_file()]
    cmd = ' '.join([shlex.quote(c) for c in cmd])
    os.environ['CLIP_AFTER_ALLREDUCE'] = str(clip_after_allreduce)
    os.environ['MAX_GLOBAL_NORM'] = str(max_global_norm)
    os.environ['GRADIENT_MERGE_STEPS'] = str(gradient_merge_steps)
    os.environ['USE_MASTER_ACC_GRAD'] = str(1 if use_master_acc_grad else 0)
    os.environ['FLAGS_dynamic_static_unified_comm'] = '0'
    os.environ.update(need_env)
    touch_file_env = 'SUCCESS_TOUCH_FILE'
    touch_file_name = os.path.join(temp_dir.name, f'distributed_fused_lamb_touch_file_{os.getpid()}')
    os.environ[touch_file_env] = touch_file_name
    try:
        assert os.system(cmd) == 0 and os.path.exists(touch_file_name), f'Test failed when {args}'
    finally:
        temp_dir.cleanup()

class TestDistributedFusedLambWithClip(unittest.TestCase):

    def test_1(self):
        if False:
            return 10
        run_test(clip_after_allreduce=True, max_global_norm=0.01)

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        run_test(clip_after_allreduce=False, max_global_norm=0.01)

    def test_1_new_comm(self):
        if False:
            print('Hello World!')
        run_test(clip_after_allreduce=True, max_global_norm=0.01, need_env={'FLAGS_dynamic_static_unified_comm': 'true'})

    def test_2_new_comm(self):
        if False:
            print('Hello World!')
        run_test(clip_after_allreduce=False, max_global_norm=0.01, need_env={'FLAGS_dynamic_static_unified_comm': 'true'})
if __name__ == '__main__':
    unittest.main()