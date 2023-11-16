import argparse
import logging
import sys
from unittest.mock import patch
import run_glue_with_pabee
from transformers.testing_utils import TestCasePlus
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def get_setup_file():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args.f

class PabeeTests(TestCasePlus):

    def test_run_glue(self):
        if False:
            while True:
                i = 10
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_glue_with_pabee.py\n            --model_type albert\n            --model_name_or_path albert-base-v2\n            --data_dir ./tests/fixtures/tests_samples/MRPC/\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --task_name mrpc\n            --do_train\n            --do_eval\n            --per_gpu_train_batch_size=2\n            --per_gpu_eval_batch_size=1\n            --learning_rate=2e-5\n            --max_steps=50\n            --warmup_steps=2\n            --seed=42\n            --max_seq_length=128\n            '.split()
        with patch.object(sys, 'argv', testargs):
            result = run_glue_with_pabee.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)