import argparse
import json
import logging
import os
import sys
from unittest import skip
from unittest.mock import patch
import tensorflow as tf
from transformers.testing_utils import TestCasePlus, get_gpu_count, slow
SRC_DIRS = [os.path.join(os.path.dirname(__file__), dirname) for dirname in ['text-generation', 'text-classification', 'token-classification', 'language-modeling', 'multiple-choice', 'question-answering', 'summarization', 'translation', 'image-classification']]
sys.path.extend(SRC_DIRS)
if SRC_DIRS is not None:
    import run_clm
    import run_image_classification
    import run_mlm
    import run_ner
    import run_qa as run_squad
    import run_summarization
    import run_swag
    import run_text_classification
    import run_translation
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

def is_cuda_available():
    if False:
        for i in range(10):
            print('nop')
    return bool(tf.config.list_physical_devices('GPU'))
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

class ExamplesTests(TestCasePlus):

    @skip('Skipping until shape inference for to_tf_dataset PR is merged.')
    def test_run_text_classification(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_text_classification.py\n            --model_name_or_path distilbert-base-uncased\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv\n            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv\n            --do_train\n            --do_eval\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --learning_rate=1e-4\n            --max_steps=10\n            --warmup_steps=2\n            --seed=42\n            --max_seq_length=128\n            '.split()
        if is_cuda_available():
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_text_classification.main()
            tf.keras.mixed_precision.set_global_policy('float32')
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)

    def test_run_clm(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_clm.py\n            --model_name_or_path distilgpt2\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --do_train\n            --do_eval\n            --block_size 128\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --num_train_epochs 2\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            '.split()
        if len(tf.config.list_physical_devices('GPU')) > 1:
            return
        with patch.object(sys, 'argv', testargs):
            run_clm.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_perplexity'], 100)

    def test_run_mlm(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_mlm.py\n            --model_name_or_path distilroberta-base\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --max_seq_length 64\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --prediction_loss_only\n            --num_train_epochs=1\n            --learning_rate=1e-4\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_mlm.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_perplexity'], 42)

    def test_run_ner(self):
        if False:
            print('Hello World!')
        epochs = 7 if get_gpu_count() > 1 else 2
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_ner.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/conll/sample.json\n            --validation_file tests/fixtures/tests_samples/conll/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --warmup_steps=2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=2\n            --num_train_epochs={epochs}\n            --seed 7\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['accuracy'], 0.75)

    def test_run_squad(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_qa.py\n            --model_name_or_path bert-base-uncased\n            --version_2_with_negative\n            --train_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=10\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_squad.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['f1'], 30)
            self.assertGreaterEqual(result['exact'], 30)

    def test_run_swag(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_swag.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/swag/sample.json\n            --validation_file tests/fixtures/tests_samples/swag/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=20\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_swag.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['val_accuracy'], 0.8)

    @slow
    def test_run_summarization(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_summarization.py\n            --model_name_or_path t5-small\n            --train_file tests/fixtures/tests_samples/xsum/sample.json\n            --validation_file tests/fixtures/tests_samples/xsum/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=50\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_summarization.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['rouge1'], 10)
            self.assertGreaterEqual(result['rouge2'], 2)
            self.assertGreaterEqual(result['rougeL'], 7)
            self.assertGreaterEqual(result['rougeLsum'], 7)

    @slow
    def test_run_translation(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_translation.py\n            --model_name_or_path Rocketknight1/student_marian_en_ro_6_1\n            --source_lang en\n            --target_lang ro\n            --train_file tests/fixtures/tests_samples/wmt16/sample.json\n            --validation_file tests/fixtures/tests_samples/wmt16/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --learning_rate=3e-3\n            --num_train_epochs 12\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --source_lang en_XX\n            --target_lang ro_RO\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['bleu'], 30)

    def test_run_image_classification(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_image_classification.py\n            --dataset_name hf-internal-testing/cats_vs_dogs_sample\n            --model_name_or_path microsoft/resnet-18\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --dataloader_num_workers 16\n            --num_train_epochs 2\n            --train_val_split 0.1\n            --seed 42\n            --ignore_mismatched_sizes True\n            '.split()
        with patch.object(sys, 'argv', testargs):
            run_image_classification.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['accuracy'], 0.7)