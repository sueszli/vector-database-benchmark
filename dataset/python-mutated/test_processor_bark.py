import os
import shutil
import tempfile
import unittest
import numpy as np
from transformers import AutoTokenizer, BarkProcessor
from transformers.testing_utils import require_torch, slow

@require_torch
class BarkProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.checkpoint = 'suno/bark-small'
        self.tmpdirname = tempfile.mkdtemp()
        self.voice_preset = 'en_speaker_1'
        self.input_string = 'This is a test string'
        self.speaker_embeddings_dict_path = 'speaker_embeddings_path.json'
        self.speaker_embeddings_directory = 'speaker_embeddings'

    def get_tokenizer(self, **kwargs):
        if False:
            while True:
                i = 10
        return AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        if False:
            return 10
        tokenizer = self.get_tokenizer()
        processor = BarkProcessor(tokenizer=tokenizer)
        processor.save_pretrained(self.tmpdirname)
        processor = BarkProcessor.from_pretrained(self.tmpdirname)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

    @slow
    def test_save_load_pretrained_additional_features(self):
        if False:
            while True:
                i = 10
        processor = BarkProcessor.from_pretrained(pretrained_processor_name_or_path=self.checkpoint, speaker_embeddings_dict_path=self.speaker_embeddings_dict_path)
        processor.save_pretrained(self.tmpdirname, speaker_embeddings_dict_path=self.speaker_embeddings_dict_path, speaker_embeddings_directory=self.speaker_embeddings_directory)
        tokenizer_add_kwargs = self.get_tokenizer(bos_token='(BOS)', eos_token='(EOS)')
        processor = BarkProcessor.from_pretrained(self.tmpdirname, self.speaker_embeddings_dict_path, bos_token='(BOS)', eos_token='(EOS)')
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

    def test_speaker_embeddings(self):
        if False:
            print('Hello World!')
        processor = BarkProcessor.from_pretrained(pretrained_processor_name_or_path=self.checkpoint, speaker_embeddings_dict_path=self.speaker_embeddings_dict_path)
        seq_len = 35
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8
        voice_preset = {'semantic_prompt': np.ones(seq_len), 'coarse_prompt': np.ones((nb_codebooks_coarse, seq_len)), 'fine_prompt': np.ones((nb_codebooks_total, seq_len))}
        inputs = processor(text=self.input_string, voice_preset=voice_preset)
        processed_voice_preset = inputs['history_prompt']
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())
        tmpfilename = os.path.join(self.tmpdirname, 'file.npz')
        np.savez(tmpfilename, **voice_preset)
        inputs = processor(text=self.input_string, voice_preset=tmpfilename)
        processed_voice_preset = inputs['history_prompt']
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())
        inputs = processor(text=self.input_string, voice_preset=self.voice_preset)

    def test_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = self.get_tokenizer()
        processor = BarkProcessor(tokenizer=tokenizer)
        encoded_processor = processor(text=self.input_string)
        encoded_tok = tokenizer(self.input_string, padding='max_length', max_length=256, add_special_tokens=False, return_attention_mask=True, return_token_type_ids=False)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key].squeeze().tolist())