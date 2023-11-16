import argparse
import logging
import sys
from unittest.mock import patch
import run_glue_deebert
from transformers.testing_utils import TestCasePlus, get_gpu_count, require_torch_non_multi_gpu, slow
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

class DeeBertTests(TestCasePlus):

    def setup(self) -> None:
        if False:
            i = 10
            return i + 15
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    def run_and_check(self, args):
        if False:
            for i in range(10):
                print('nop')
        n_gpu = get_gpu_count()
        if n_gpu > 1:
            pass
        else:
            args.insert(0, 'run_glue_deebert.py')
            with patch.object(sys, 'argv', args):
                result = run_glue_deebert.main()
                for value in result.values():
                    self.assertGreaterEqual(value, 0.666)

    @slow
    @require_torch_non_multi_gpu
    def test_glue_deebert_train(self):
        if False:
            print('Hello World!')
        train_args = '\n            --model_type roberta\n            --model_name_or_path roberta-base\n            --task_name MRPC\n            --do_train\n            --do_eval\n            --do_lower_case\n            --data_dir ./tests/fixtures/tests_samples/MRPC/\n            --max_seq_length 128\n            --per_gpu_eval_batch_size=1\n            --per_gpu_train_batch_size=8\n            --learning_rate 2e-4\n            --num_train_epochs 3\n            --overwrite_output_dir\n            --seed 42\n            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage\n            --plot_data_dir ./examples/deebert/results/\n            --save_steps 0\n            --overwrite_cache\n            --eval_after_first_stage\n            '.split()
        self.run_and_check(train_args)
        eval_args = '\n            --model_type roberta\n            --model_name_or_path ./examples/deebert/saved_models/roberta-base/MRPC/two_stage\n            --task_name MRPC\n            --do_eval\n            --do_lower_case\n            --data_dir ./tests/fixtures/tests_samples/MRPC/\n            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage\n            --plot_data_dir ./examples/deebert/results/\n            --max_seq_length 128\n            --eval_each_highway\n            --eval_highway\n            --overwrite_cache\n            --per_gpu_eval_batch_size=1\n            '.split()
        self.run_and_check(eval_args)
        entropy_eval_args = '\n            --model_type roberta\n            --model_name_or_path ./examples/deebert/saved_models/roberta-base/MRPC/two_stage\n            --task_name MRPC\n            --do_eval\n            --do_lower_case\n            --data_dir ./tests/fixtures/tests_samples/MRPC/\n            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage\n            --plot_data_dir ./examples/deebert/results/\n            --max_seq_length 128\n            --early_exit_entropy 0.1\n            --eval_highway\n            --overwrite_cache\n            --per_gpu_eval_batch_size=1\n            '.split()
        self.run_and_check(entropy_eval_args)