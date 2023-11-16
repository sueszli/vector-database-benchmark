import argparse
import json
import logging
import os
import sys
from unittest.mock import patch
from transformers.testing_utils import TestCasePlus, get_gpu_count, slow
SRC_DIRS = [os.path.join(os.path.dirname(__file__), dirname) for dirname in ['text-classification', 'language-modeling', 'summarization', 'token-classification', 'question-answering', 'speech-recognition']]
sys.path.extend(SRC_DIRS)
if SRC_DIRS is not None:
    import run_clm_flax
    import run_flax_glue
    import run_flax_ner
    import run_flax_speech_recognition_seq2seq
    import run_mlm_flax
    import run_qa
    import run_summarization_flax
    import run_t5_mlm_flax
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def get_setup_file():
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args.f

def get_results(output_dir, split='eval'):
    if False:
        for i in range(10):
            print('nop')
    path = os.path.join(output_dir, f'{split}_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    raise ValueError(f"can't find {path}")
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

class ExamplesTests(TestCasePlus):

    def test_run_glue(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_glue.py\n            --model_name_or_path distilbert-base-uncased\n            --output_dir {tmp_dir}\n            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv\n            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --learning_rate=1e-4\n            --eval_steps=2\n            --warmup_steps=2\n            --seed=42\n            --max_seq_length=128\n            '.split()
        with patch.object(sys, 'argv', testargs):
            run_flax_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)

    @slow
    def test_run_clm(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_clm_flax.py\n            --model_name_or_path distilgpt2\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --do_train\n            --do_eval\n            --block_size 128\n            --per_device_train_batch_size 4\n            --per_device_eval_batch_size 4\n            --num_train_epochs 2\n            --logging_steps 2 --eval_steps 2\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            '.split()
        with patch.object(sys, 'argv', testargs):
            run_clm_flax.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_perplexity'], 100)

    @slow
    def test_run_summarization(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_summarization.py\n            --model_name_or_path t5-small\n            --train_file tests/fixtures/tests_samples/xsum/sample.json\n            --validation_file tests/fixtures/tests_samples/xsum/sample.json\n            --test_file tests/fixtures/tests_samples/xsum/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --num_train_epochs=3\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --do_predict\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --predict_with_generate\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_summarization_flax.main()
            result = get_results(tmp_dir, split='test')
            self.assertGreaterEqual(result['test_rouge1'], 10)
            self.assertGreaterEqual(result['test_rouge2'], 2)
            self.assertGreaterEqual(result['test_rougeL'], 7)
            self.assertGreaterEqual(result['test_rougeLsum'], 7)

    @slow
    def test_run_mlm(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_mlm.py\n            --model_name_or_path distilroberta-base\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_seq_length 128\n            --per_device_train_batch_size 4\n            --per_device_eval_batch_size 4\n            --logging_steps 2 --eval_steps 2\n            --do_train\n            --do_eval\n            --num_train_epochs=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_mlm_flax.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_perplexity'], 42)

    @slow
    def test_run_t5_mlm(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_t5_mlm_flax.py\n            --model_name_or_path t5-small\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --do_train\n            --do_eval\n            --max_seq_length 128\n            --per_device_train_batch_size 4\n            --per_device_eval_batch_size 4\n            --num_train_epochs 2\n            --logging_steps 2 --eval_steps 2\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            '.split()
        with patch.object(sys, 'argv', testargs):
            run_t5_mlm_flax.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.42)

    @slow
    def test_run_ner(self):
        if False:
            for i in range(10):
                print('nop')
        epochs = 7 if get_gpu_count() > 1 else 2
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_flax_ner.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/conll/sample.json\n            --validation_file tests/fixtures/tests_samples/conll/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --warmup_steps=2\n            --learning_rate=2e-4\n            --logging_steps 2 --eval_steps 2\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=2\n            --num_train_epochs={epochs}\n            --seed 7\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_flax_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)
            self.assertGreaterEqual(result['eval_f1'], 0.3)

    @slow
    def test_run_qa(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_qa.py\n            --model_name_or_path bert-base-uncased\n            --version_2_with_negative\n            --train_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --num_train_epochs=3\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --logging_steps 2 --eval_steps 2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_qa.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_f1'], 30)
            self.assertGreaterEqual(result['eval_exact'], 30)

    @slow
    def test_run_flax_speech_recognition_seq2seq(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_flax_speech_recognition_seq2seq.py\n            --model_name_or_path openai/whisper-tiny.en\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config clean\n            --train_split_name validation\n            --eval_split_name validation\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --num_train_epochs=2\n            --max_train_samples 10\n            --max_eval_samples 10\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --predict_with_generate\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_flax_speech_recognition_seq2seq.main()
            result = get_results(tmp_dir, split='eval')
            self.assertLessEqual(result['eval_wer'], 0.05)