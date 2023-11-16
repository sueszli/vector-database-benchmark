import os
import sys
from transformers.testing_utils import TestCasePlus, execute_subprocess_async, get_gpu_count, require_torch_gpu, slow
from .utils import load_json

class TestSummarizationDistillerMultiGPU(TestCasePlus):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        return cls

    @slow
    @require_torch_gpu
    def test_distributed_eval(self):
        if False:
            print('Hello World!')
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'\n            --model_name Helsinki-NLP/opus-mt-en-ro\n            --save_dir {output_dir}\n            --data_dir {self.test_file_dir_str}/test_data/wmt_en_ro\n            --num_beams 2\n            --task translation\n        '.split()
        n_gpu = get_gpu_count()
        distributed_args = f'\n            -m torch.distributed.launch\n            --nproc_per_node={n_gpu}\n            {self.test_file_dir}/run_distributed_eval.py\n        '.split()
        cmd = [sys.executable] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())
        metrics_save_path = os.path.join(output_dir, 'test_bleu.json')
        metrics = load_json(metrics_save_path)
        self.assertGreaterEqual(metrics['bleu'], 25)