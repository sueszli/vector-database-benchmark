import json
import logging
import os
import sys
from time import time
from unittest.mock import patch
from transformers.testing_utils import TestCasePlus, require_torch_tpu
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def get_results(output_dir):
    if False:
        print('Hello World!')
    results = {}
    path = os.path.join(output_dir, 'all_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

@require_torch_tpu
class TorchXLAExamplesTests(TestCasePlus):

    def test_run_glue(self):
        if False:
            return 10
        import xla_spawn
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            ./examples/pytorch/text-classification/run_glue.py\n            --num_cores=8\n            ./examples/pytorch/text-classification/run_glue.py\n            --model_name_or_path distilbert-base-uncased\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv\n            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv\n            --do_train\n            --do_eval\n            --debug tpu_metrics_debug\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --learning_rate=1e-4\n            --max_steps=10\n            --warmup_steps=2\n            --seed=42\n            --max_seq_length=128\n            '.split()
        with patch.object(sys, 'argv', testargs):
            start = time()
            xla_spawn.main()
            end = time()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)
            self.assertLess(end - start, 500)

    def test_trainer_tpu(self):
        if False:
            for i in range(10):
                print('nop')
        import xla_spawn
        testargs = '\n            ./tests/test_trainer_tpu.py\n            --num_cores=8\n            ./tests/test_trainer_tpu.py\n            '.split()
        with patch.object(sys, 'argv', testargs):
            xla_spawn.main()