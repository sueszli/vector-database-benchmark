import io
import json
import unittest
from parameterized import parameterized
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers.testing_utils import get_tests_dir, require_torch, slow, torch_device
from utils import calculate_bleu
filename = get_tests_dir() + '/test_data/fsmt/fsmt_val_data.json'
with io.open(filename, 'r', encoding='utf-8') as f:
    bleu_data = json.load(f)

@require_torch
class ModelEvalTester(unittest.TestCase):

    def get_tokenizer(self, mname):
        if False:
            print('Hello World!')
        return FSMTTokenizer.from_pretrained(mname)

    def get_model(self, mname):
        if False:
            print('Hello World!')
        model = FSMTForConditionalGeneration.from_pretrained(mname).to(torch_device)
        if torch_device == 'cuda':
            model.half()
        return model

    @parameterized.expand([['en-ru', 26.0], ['ru-en', 22.0], ['en-de', 22.0], ['de-en', 29.0]])
    @slow
    def test_bleu_scores(self, pair, min_bleu_score):
        if False:
            print('Hello World!')
        mname = f'facebook/wmt19-{pair}'
        tokenizer = self.get_tokenizer(mname)
        model = self.get_model(mname)
        src_sentences = bleu_data[pair]['src']
        tgt_sentences = bleu_data[pair]['tgt']
        batch = tokenizer(src_sentences, return_tensors='pt', truncation=True, padding='longest').to(torch_device)
        outputs = model.generate(input_ids=batch.input_ids, num_beams=8)
        decoded_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        scores = calculate_bleu(decoded_sentences, tgt_sentences)
        print(scores)
        self.assertGreaterEqual(scores['bleu'], min_bleu_score)