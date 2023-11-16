import unittest
from queue import Empty
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer, TextStreamer, is_torch_available
from transformers.testing_utils import CaptureStdout, require_torch, torch_device
from ..test_modeling_common import ids_tensor
if is_torch_available():
    import torch
    from transformers import AutoModelForCausalLM

@require_torch
class StreamerTester(unittest.TestCase):

    def test_text_streamer_matches_non_streaming(self):
        if False:
            print('Hello World!')
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2').to(torch_device)
        model.config.eos_token_id = -1
        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])
        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        streamer_text = cs.out[:-1]
        self.assertEqual(streamer_text, greedy_text)

    def test_iterator_streamer_matches_non_streaming(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2').to(torch_device)
        model.config.eos_token_id = -1
        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {'input_ids': input_ids, 'max_new_tokens': 10, 'do_sample': False, 'streamer': streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ''
        for new_text in streamer:
            streamer_text += new_text
        self.assertEqual(streamer_text, greedy_text)

    def test_text_streamer_skip_prompt(self):
        if False:
            print('Hello World!')
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2').to(torch_device)
        model.config.eos_token_id = -1
        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        new_greedy_ids = greedy_ids[:, input_ids.shape[1]:]
        new_greedy_text = tokenizer.decode(new_greedy_ids[0])
        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        streamer_text = cs.out[:-1]
        self.assertEqual(streamer_text, new_greedy_text)

    def test_text_streamer_decode_kwargs(self):
        if False:
            return 10
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(torch_device)
        model.config.eos_token_id = -1
        input_ids = torch.ones((1, 5), device=torch_device).long() * model.config.bos_token_id
        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
            model.generate(input_ids, max_new_tokens=1, do_sample=False, streamer=streamer)
        streamer_text = cs.out[:-1]
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors='pt')
        self.assertEqual(streamer_text_tokenized.input_ids.shape, (1, 1))

    def test_iterator_streamer_timeout(self):
        if False:
            return 10
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2').to(torch_device)
        model.config.eos_token_id = -1
        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {'input_ids': input_ids, 'max_new_tokens': 10, 'do_sample': False, 'streamer': streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        with self.assertRaises(Empty):
            streamer_text = ''
            for new_text in streamer:
                streamer_text += new_text