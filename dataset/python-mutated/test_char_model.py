"""
Currently tests a few configurations of files for creating a charlm vocab

Also has a skeleton test of loading & saving a charlm
"""
from collections import Counter
import glob
import lzma
import os
import tempfile
import pytest
from stanza.models import charlm
from stanza.models.common import char_model
from stanza.tests import TEST_MODELS_DIR
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
fake_text_1 = '\nUnban mox opal!\nI hate watching Peppa Pig\n'
fake_text_2 = '\nThis is plastic cheese\n'

class TestCharModel:

    def test_single_file_vocab(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, 'text.txt')
            with open(sample_file, 'w', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(sample_file)
        for i in fake_text_1:
            assert i in vocab
        assert 'Q' not in vocab

    def test_single_file_xz_vocab(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, 'text.txt.xz')
            with lzma.open(sample_file, 'wt', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(sample_file)
        for i in fake_text_1:
            assert i in vocab
        assert 'Q' not in vocab

    def test_single_file_dir_vocab(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, 'text.txt')
            with open(sample_file, 'w', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(tempdir)
        for i in fake_text_1:
            assert i in vocab
        assert 'Q' not in vocab

    def test_multiple_files_vocab(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, 't1.txt')
            with open(sample_file, 'w', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            sample_file = os.path.join(tempdir, 't2.txt.xz')
            with lzma.open(sample_file, 'wt', encoding='utf-8') as fout:
                fout.write(fake_text_2)
            vocab = char_model.build_charlm_vocab(tempdir)
        for i in fake_text_1:
            assert i in vocab
        for i in fake_text_2:
            assert i in vocab
        assert 'Q' not in vocab

    def test_cutoff_vocab(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, 't1.txt')
            with open(sample_file, 'w', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            sample_file = os.path.join(tempdir, 't2.txt.xz')
            with lzma.open(sample_file, 'wt', encoding='utf-8') as fout:
                fout.write(fake_text_2)
            vocab = char_model.build_charlm_vocab(tempdir, cutoff=2)
        counts = Counter(fake_text_1) + Counter(fake_text_2)
        for (letter, count) in counts.most_common():
            if count < 2:
                assert letter not in vocab
            else:
                assert letter in vocab

    def test_build_model(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the whole thing on a small dataset for an iteration or two\n        '
        with tempfile.TemporaryDirectory() as tempdir:
            eval_file = os.path.join(tempdir, 'en_test.dev.txt')
            with open(eval_file, 'w', encoding='utf-8') as fout:
                fout.write(fake_text_1)
            train_file = os.path.join(tempdir, 'en_test.train.txt')
            with open(train_file, 'w', encoding='utf-8') as fout:
                for i in range(1000):
                    fout.write(fake_text_1)
                    fout.write('\n')
                    fout.write(fake_text_2)
                    fout.write('\n')
            save_name = 'en_test.forward.pt'
            vocab_save_name = 'en_text.vocab.pt'
            checkpoint_save_name = 'en_text.checkpoint.pt'
            args = ['--train_file', train_file, '--eval_file', eval_file, '--eval_steps', '0', '--epochs', '2', '--cutoff', '1', '--batch_size', '%d' % len(fake_text_1), '--shorthand', 'en_test', '--save_dir', tempdir, '--save_name', save_name, '--vocab_save_name', vocab_save_name, '--checkpoint_save_name', checkpoint_save_name]
            args = charlm.parse_args(args)
            charlm.train(args)
            assert os.path.exists(os.path.join(tempdir, vocab_save_name))
            assert os.path.exists(os.path.join(tempdir, save_name))
            model = char_model.CharacterLanguageModel.load(os.path.join(tempdir, save_name))
            assert os.path.exists(os.path.join(tempdir, checkpoint_save_name))
            model = char_model.CharacterLanguageModel.load(os.path.join(tempdir, checkpoint_save_name))
            trainer = char_model.CharacterLanguageModelTrainer.load(args, os.path.join(tempdir, checkpoint_save_name))
            assert trainer.global_step > 0
            assert trainer.epoch == 2
            charlm.get_current_lr(trainer, args)
            vocab = charlm.load_char_vocab(os.path.join(tempdir, vocab_save_name))
            trainer = char_model.CharacterLanguageModelTrainer.from_new_model(args, vocab)
            assert charlm.get_current_lr(trainer, args) == args['lr0']

    @pytest.fixture(scope='class')
    def english_forward(self):
        if False:
            for i in range(10):
                print('nop')
        models_path = os.path.join(TEST_MODELS_DIR, 'en', 'forward_charlm', '*')
        models = glob.glob(models_path)
        assert len(models) >= 1
        model_file = models[0]
        return char_model.CharacterLanguageModel.load(model_file)

    @pytest.fixture(scope='class')
    def english_backward(self):
        if False:
            print('Hello World!')
        models_path = os.path.join(TEST_MODELS_DIR, 'en', 'backward_charlm', '*')
        models = glob.glob(models_path)
        assert len(models) >= 1
        model_file = models[0]
        return char_model.CharacterLanguageModel.load(model_file)

    def test_load_model(self, english_forward, english_backward):
        if False:
            print('Hello World!')
        '\n        Check that basic loading functions work\n        '
        assert english_forward.is_forward_lm
        assert not english_backward.is_forward_lm

    def test_save_load_model(self, english_forward, english_backward):
        if False:
            return 10
        '\n        Load, save, and load again\n        '
        with tempfile.TemporaryDirectory() as tempdir:
            for model in (english_forward, english_backward):
                save_file = os.path.join(tempdir, 'resaved', 'charlm.pt')
                model.save(save_file)
                reloaded = char_model.CharacterLanguageModel.load(save_file)
                assert model.is_forward_lm == reloaded.is_forward_lm