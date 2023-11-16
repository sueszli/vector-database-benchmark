import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from accelerate.utils import write_basic_config
from transformers.testing_utils import TestCasePlus, backend_device_count, run_command, slow, torch_device
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def get_setup_file():
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args.f

def get_results(output_dir):
    if False:
        while True:
            i = 10
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

class ExamplesTestsNoTrainer(TestCasePlus):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls.tmpdir, 'default_config.yml')
        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ['accelerate', 'launch', '--config_file', cls.configPath]

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(cls.tmpdir)

    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_glue_no_trainer(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/text-classification/run_glue_no_trainer.py\n            --model_name_or_path distilbert-base-uncased\n            --output_dir {tmp_dir}\n            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv\n            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --learning_rate=1e-4\n            --seed=42\n            --num_warmup_steps=2\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_accuracy'], 0.75)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'glue_no_trainer')))

    @unittest.skip('Zach is working on this.')
    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_clm_no_trainer(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/language-modeling/run_clm_no_trainer.py\n            --model_name_or_path distilgpt2\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --block_size 128\n            --per_device_train_batch_size 5\n            --per_device_eval_batch_size 5\n            --num_train_epochs 2\n            --output_dir {tmp_dir}\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        if backend_device_count(torch_device) > 1:
            return
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertLess(result['perplexity'], 100)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'clm_no_trainer')))

    @unittest.skip('Zach is working on this.')
    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_mlm_no_trainer(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/language-modeling/run_mlm_no_trainer.py\n            --model_name_or_path distilroberta-base\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --output_dir {tmp_dir}\n            --num_train_epochs=1\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertLess(result['perplexity'], 42)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'mlm_no_trainer')))

    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_ner_no_trainer(self):
        if False:
            return 10
        epochs = 7 if backend_device_count(torch_device) > 1 else 2
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/token-classification/run_ner_no_trainer.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/conll/sample.json\n            --validation_file tests/fixtures/tests_samples/conll/sample.json\n            --output_dir {tmp_dir}\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=2\n            --num_train_epochs={epochs}\n            --seed 7\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_accuracy'], 0.75)
        self.assertLess(result['train_loss'], 0.6)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'ner_no_trainer')))

    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_squad_no_trainer(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/question-answering/run_qa_no_trainer.py\n            --model_name_or_path bert-base-uncased\n            --version_2_with_negative\n            --train_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --output_dir {tmp_dir}\n            --seed=42\n            --max_train_steps=10\n            --num_warmup_steps=2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_f1'], 28)
        self.assertGreaterEqual(result['eval_exact'], 28)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'qa_no_trainer')))

    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_swag_no_trainer(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/multiple-choice/run_swag_no_trainer.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/swag/sample.json\n            --validation_file tests/fixtures/tests_samples/swag/sample.json\n            --output_dir {tmp_dir}\n            --max_train_steps=20\n            --num_warmup_steps=2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_accuracy'], 0.8)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'swag_no_trainer')))

    @slow
    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_summarization_no_trainer(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/summarization/run_summarization_no_trainer.py\n            --model_name_or_path t5-small\n            --train_file tests/fixtures/tests_samples/xsum/sample.json\n            --validation_file tests/fixtures/tests_samples/xsum/sample.json\n            --output_dir {tmp_dir}\n            --max_train_steps=50\n            --num_warmup_steps=8\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_rouge1'], 10)
        self.assertGreaterEqual(result['eval_rouge2'], 2)
        self.assertGreaterEqual(result['eval_rougeL'], 7)
        self.assertGreaterEqual(result['eval_rougeLsum'], 7)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'summarization_no_trainer')))

    @slow
    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_translation_no_trainer(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/translation/run_translation_no_trainer.py\n            --model_name_or_path sshleifer/student_marian_en_ro_6_1\n            --source_lang en\n            --target_lang ro\n            --train_file tests/fixtures/tests_samples/wmt16/sample.json\n            --validation_file tests/fixtures/tests_samples/wmt16/sample.json\n            --output_dir {tmp_dir}\n            --max_train_steps=50\n            --num_warmup_steps=8\n            --num_beams=6\n            --learning_rate=3e-3\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --source_lang en_XX\n            --target_lang ro_RO\n            --checkpointing_steps epoch\n            --with_tracking\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_bleu'], 30)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'epoch_0')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'translation_no_trainer')))

    @slow
    def test_run_semantic_segmentation_no_trainer(self):
        if False:
            for i in range(10):
                print('nop')
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py\n            --dataset_name huggingface/semantic-segmentation-test-sample\n            --output_dir {tmp_dir}\n            --max_train_steps=10\n            --num_warmup_steps=2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --checkpointing_steps epoch\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_overall_accuracy'], 0.1)

    @mock.patch.dict(os.environ, {'WANDB_MODE': 'offline'})
    def test_run_image_classification_no_trainer(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            {self.examples_dir}/pytorch/image-classification/run_image_classification_no_trainer.py\n            --model_name_or_path google/vit-base-patch16-224-in21k\n            --dataset_name hf-internal-testing/cats_vs_dogs_sample\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --max_train_steps 2\n            --train_val_split 0.1\n            --seed 42\n            --output_dir {tmp_dir}\n            --with_tracking\n            --checkpointing_steps 1\n        '.split()
        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result['eval_accuracy'], 0.4)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'step_1')))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'image_classification_no_trainer')))