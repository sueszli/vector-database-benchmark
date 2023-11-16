import json
import logging
import os
import sys
from unittest.mock import patch
from transformers import ViTMAEForPreTraining, Wav2Vec2ForPreTraining
from transformers.testing_utils import CaptureLogger, TestCasePlus, backend_device_count, is_torch_fp16_available_on_device, slow, torch_device
SRC_DIRS = [os.path.join(os.path.dirname(__file__), dirname) for dirname in ['text-generation', 'text-classification', 'token-classification', 'language-modeling', 'multiple-choice', 'question-answering', 'summarization', 'translation', 'image-classification', 'speech-recognition', 'audio-classification', 'speech-pretraining', 'image-pretraining', 'semantic-segmentation']]
sys.path.extend(SRC_DIRS)
if SRC_DIRS is not None:
    import run_audio_classification
    import run_clm
    import run_generation
    import run_glue
    import run_image_classification
    import run_mae
    import run_mlm
    import run_ner
    import run_qa as run_squad
    import run_semantic_segmentation
    import run_seq2seq_qa as run_squad_seq2seq
    import run_speech_recognition_ctc
    import run_speech_recognition_ctc_adapter
    import run_speech_recognition_seq2seq
    import run_summarization
    import run_swag
    import run_translation
    import run_wav2vec2_pretraining_no_trainer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

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

class ExamplesTests(TestCasePlus):

    def test_run_glue(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_glue.py\n            --model_name_or_path distilbert-base-uncased\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv\n            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv\n            --do_train\n            --do_eval\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --learning_rate=1e-4\n            --max_steps=10\n            --warmup_steps=2\n            --seed=42\n            --max_seq_length=128\n            '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)

    def test_run_clm(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_clm.py\n            --model_name_or_path distilgpt2\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --do_train\n            --do_eval\n            --block_size 128\n            --per_device_train_batch_size 5\n            --per_device_eval_batch_size 5\n            --num_train_epochs 2\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            '.split()
        if backend_device_count(torch_device) > 1:
            return
        if torch_device == 'cpu':
            testargs.append('--use_cpu')
        with patch.object(sys, 'argv', testargs):
            run_clm.main()
            result = get_results(tmp_dir)
            self.assertLess(result['perplexity'], 100)

    def test_run_clm_config_overrides(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_clm.py\n            --model_type gpt2\n            --tokenizer_name gpt2\n            --train_file ./tests/fixtures/sample_text.txt\n            --output_dir {tmp_dir}\n            --config_overrides n_embd=10,n_head=2\n            '.split()
        if torch_device == 'cpu':
            testargs.append('--use_cpu')
        logger = run_clm.logger
        with patch.object(sys, 'argv', testargs):
            with CaptureLogger(logger) as cl:
                run_clm.main()
        self.assertIn('"n_embd": 10', cl.out)
        self.assertIn('"n_head": 2', cl.out)

    def test_run_mlm(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_mlm.py\n            --model_name_or_path distilroberta-base\n            --train_file ./tests/fixtures/sample_text.txt\n            --validation_file ./tests/fixtures/sample_text.txt\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --prediction_loss_only\n            --num_train_epochs=1\n        '.split()
        if torch_device == 'cpu':
            testargs.append('--use_cpu')
        with patch.object(sys, 'argv', testargs):
            run_mlm.main()
            result = get_results(tmp_dir)
            self.assertLess(result['perplexity'], 42)

    def test_run_ner(self):
        if False:
            i = 10
            return i + 15
        epochs = 7 if backend_device_count(torch_device) > 1 else 2
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_ner.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/conll/sample.json\n            --validation_file tests/fixtures/tests_samples/conll/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --warmup_steps=2\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=2\n            --num_train_epochs={epochs}\n            --seed 7\n        '.split()
        if torch_device == 'cpu':
            testargs.append('--use_cpu')
        with patch.object(sys, 'argv', testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.75)
            self.assertLess(result['eval_loss'], 0.5)

    def test_run_squad(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_qa.py\n            --model_name_or_path bert-base-uncased\n            --version_2_with_negative\n            --train_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=10\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_squad.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_f1'], 30)
            self.assertGreaterEqual(result['eval_exact'], 30)

    def test_run_squad_seq2seq(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_seq2seq_qa.py\n            --model_name_or_path t5-small\n            --context_column context\n            --question_column question\n            --answer_column answers\n            --version_2_with_negative\n            --train_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=10\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --predict_with_generate\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_squad_seq2seq.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_f1'], 30)
            self.assertGreaterEqual(result['eval_exact'], 30)

    def test_run_swag(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_swag.py\n            --model_name_or_path bert-base-uncased\n            --train_file tests/fixtures/tests_samples/swag/sample.json\n            --validation_file tests/fixtures/tests_samples/swag/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=20\n            --warmup_steps=2\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_swag.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.8)

    def test_generation(self):
        if False:
            print('Hello World!')
        testargs = ['run_generation.py', '--prompt=Hello', '--length=10', '--seed=42']
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        (model_type, model_name) = ('--model_type=gpt2', '--model_name_or_path=sshleifer/tiny-gpt2')
        with patch.object(sys, 'argv', testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)

    @slow
    def test_run_summarization(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_summarization.py\n            --model_name_or_path t5-small\n            --train_file tests/fixtures/tests_samples/xsum/sample.json\n            --validation_file tests/fixtures/tests_samples/xsum/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=50\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --predict_with_generate\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_summarization.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_rouge1'], 10)
            self.assertGreaterEqual(result['eval_rouge2'], 2)
            self.assertGreaterEqual(result['eval_rougeL'], 7)
            self.assertGreaterEqual(result['eval_rougeLsum'], 7)

    @slow
    def test_run_translation(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_translation.py\n            --model_name_or_path sshleifer/student_marian_en_ro_6_1\n            --source_lang en\n            --target_lang ro\n            --train_file tests/fixtures/tests_samples/wmt16/sample.json\n            --validation_file tests/fixtures/tests_samples/wmt16/sample.json\n            --output_dir {tmp_dir}\n            --overwrite_output_dir\n            --max_steps=50\n            --warmup_steps=8\n            --do_train\n            --do_eval\n            --learning_rate=3e-3\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --predict_with_generate\n            --source_lang en_XX\n            --target_lang ro_RO\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_bleu'], 30)

    def test_run_image_classification(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_image_classification.py\n            --output_dir {tmp_dir}\n            --model_name_or_path google/vit-base-patch16-224-in21k\n            --dataset_name hf-internal-testing/cats_vs_dogs_sample\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --dataloader_num_workers 16\n            --metric_for_best_model accuracy\n            --max_steps 10\n            --train_val_split 0.1\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_image_classification.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_accuracy'], 0.8)

    def test_run_speech_recognition_ctc(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_speech_recognition_ctc.py\n            --output_dir {tmp_dir}\n            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config_name clean\n            --train_split_name validation\n            --eval_split_name validation\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --preprocessing_num_workers 16\n            --max_steps 10\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_speech_recognition_ctc.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_loss'], result['train_loss'])

    def test_run_speech_recognition_ctc_adapter(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_speech_recognition_ctc_adapter.py\n            --output_dir {tmp_dir}\n            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config_name clean\n            --train_split_name validation\n            --eval_split_name validation\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --preprocessing_num_workers 16\n            --max_steps 10\n            --target_language tur\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_speech_recognition_ctc_adapter.main()
            result = get_results(tmp_dir)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, './adapter.tur.safetensors')))
            self.assertLess(result['eval_loss'], result['train_loss'])

    def test_run_speech_recognition_seq2seq(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_speech_recognition_seq2seq.py\n            --output_dir {tmp_dir}\n            --model_name_or_path hf-internal-testing/tiny-random-speech-encoder-decoder\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config_name clean\n            --train_split_name validation\n            --eval_split_name validation\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 4\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --preprocessing_num_workers 16\n            --max_steps 10\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_speech_recognition_seq2seq.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_loss'], result['train_loss'])

    def test_run_audio_classification(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_audio_classification.py\n            --output_dir {tmp_dir}\n            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2\n            --dataset_name anton-l/superb_demo\n            --dataset_config_name ks\n            --train_split_name test\n            --eval_split_name test\n            --audio_column_name audio\n            --label_column_name label\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --num_train_epochs 10\n            --max_steps 50\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_audio_classification.main()
            result = get_results(tmp_dir)
            self.assertLess(result['eval_loss'], result['train_loss'])

    def test_run_wav2vec2_pretraining(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_wav2vec2_pretraining_no_trainer.py\n            --output_dir {tmp_dir}\n            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config_names clean\n            --dataset_split_names validation\n            --learning_rate 1e-4\n            --per_device_train_batch_size 4\n            --per_device_eval_batch_size 4\n            --preprocessing_num_workers 16\n            --max_train_steps 2\n            --validation_split_percentage 5\n            --seed 42\n        '.split()
        with patch.object(sys, 'argv', testargs):
            run_wav2vec2_pretraining_no_trainer.main()
            model = Wav2Vec2ForPreTraining.from_pretrained(tmp_dir)
            self.assertIsNotNone(model)

    def test_run_vit_mae_pretraining(self):
        if False:
            print('Hello World!')
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_mae.py\n            --output_dir {tmp_dir}\n            --dataset_name hf-internal-testing/cats_vs_dogs_sample\n            --do_train\n            --do_eval\n            --learning_rate 1e-4\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 1\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --dataloader_num_workers 16\n            --metric_for_best_model accuracy\n            --max_steps 10\n            --train_val_split 0.1\n            --seed 42\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_mae.main()
            model = ViTMAEForPreTraining.from_pretrained(tmp_dir)
            self.assertIsNotNone(model)

    def test_run_semantic_segmentation(self):
        if False:
            return 10
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f'\n            run_semantic_segmentation.py\n            --output_dir {tmp_dir}\n            --dataset_name huggingface/semantic-segmentation-test-sample\n            --do_train\n            --do_eval\n            --remove_unused_columns False\n            --overwrite_output_dir True\n            --max_steps 10\n            --learning_rate=2e-4\n            --per_device_train_batch_size=2\n            --per_device_eval_batch_size=1\n            --seed 32\n        '.split()
        if is_torch_fp16_available_on_device(torch_device):
            testargs.append('--fp16')
        with patch.object(sys, 'argv', testargs):
            run_semantic_segmentation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result['eval_overall_accuracy'], 0.1)