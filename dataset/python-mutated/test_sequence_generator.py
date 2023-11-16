import argparse
import math
import tempfile
import unittest
import numpy as np
import torch
import tests.utils as test_utils
from fairseq import search
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerModel
from fairseq.ngram_repeat_block import NGramRepeatBlock
from fairseq.sequence_generator import EnsembleModel, SequenceGenerator
from fairseq.tasks.fairseq_task import LegacyFairseqTask
DEFAULT_TEST_VOCAB_SIZE = 100

class DummyTask(LegacyFairseqTask):

    def __init__(self, args):
        if False:
            return 10
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, 'ctc', False):
            self.dictionary.add_symbol('<ctc_blank>')
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary

    @property
    def source_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.src_dict

    @property
    def target_dictionary(self):
        if False:
            i = 10
            return i + 15
        return self.dictionary

def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    if False:
        while True:
            i = 10
    dummy_dict = Dictionary()
    for (id, _) in enumerate(range(vocab_size)):
        dummy_dict.add_symbol('{}'.format(id), n=1000)
    return dummy_dict

def get_dummy_task_and_parser():
    if False:
        while True:
            i = 10
    '\n    to build a fariseq model, we need some dummy parse and task. This function\n    is used to create dummy task and parser to faciliate model/criterion test\n\n    Note: we use FbSpeechRecognitionTask as the dummy task. You may want\n    to use other task by providing another function\n    '
    parser = argparse.ArgumentParser(description='test_dummy_s2s_task', argument_default=argparse.SUPPRESS)
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return (task, parser)

class TestJitSequenceGeneratorBase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        (self.task, self.parser) = get_dummy_task_and_parser()
        eos = self.task.tgt_dict.eos()
        src_tokens = torch.randint(3, 50, (2, 10)).long()
        src_tokens = torch.cat((src_tokens, torch.LongTensor([[eos], [eos]])), -1)
        src_lengths = torch.LongTensor([2, 10])
        self.sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}
        TransformerModel.add_args(self.parser)
        args = self.parser.parse_args([])
        args.encoder_layers = 2
        args.decoder_layers = 1
        self.transformer_model = TransformerModel.build_model(args, self.task)

    def assertOutputEqual(self, hypo, pos_probs):
        if False:
            for i in range(10):
                print('nop')
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertTensorSizeEqual(hypo['positional_scores'], pos_scores)
        self.assertTensorSizeEqual(pos_scores.numel(), hypo['tokens'].numel())

    def assertTensorSizeEqual(self, t1, t2):
        if False:
            print('Hello World!')
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')

    def assertAlmostEqual(self, t1, t2):
        if False:
            print('Hello World!')
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        self.assertLess((t1 - t2).abs().max(), 0.0001)

    def assertTensorEqual(self, t1, t2):
        if False:
            while True:
                i = 10
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        self.assertEqual(t1.ne(t2).long().sum(), 0)

    def assertHypoEqual(self, h1, h2):
        if False:
            while True:
                i = 10
        'Check two hypos are equal'
        self.assertTensorEqual(h1['tokens'], h2['tokens'])
        self.assertAlmostEqual(h1['positional_scores'], h2['positional_scores'])
        self.assertLess(abs(h1['score'] - h2['score']), 1e-06)
        self.assertAlmostEqual(h1['attention'], h2['attention'])

    def _test_save_and_load(self, scripted_module):
        if False:
            return 10
        with tempfile.NamedTemporaryFile() as f:
            scripted_module.save(f.name)
            torch.jit.load(f.name)
JIT_MSG = 'Targeting OSS scriptability for the 1.6 release'

@unittest.skipIf(torch.__version__ < '1.6.0', JIT_MSG)
class TestJitSequenceGenerator(TestJitSequenceGeneratorBase):

    def test_export_transformer(self):
        if False:
            return 10
        model = self.transformer_model
        torch.jit.script(model)

    def test_ensemble_sequence_generator(self):
        if False:
            print('Hello World!')
        model = self.transformer_model
        generator = SequenceGenerator([model], self.task.tgt_dict, beam_size=2, no_repeat_ngram_size=2, max_len_b=10)
        scripted_model = torch.jit.script(generator)
        self._test_save_and_load(scripted_model)

    def test_export_ensemble_model(self):
        if False:
            i = 10
            return i + 15
        model = self.transformer_model
        ensemble_models = EnsembleModel([model])
        torch.jit.script(ensemble_models)

class TestExportSearch(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        (task, _) = get_dummy_task_and_parser()
        self.tgt_dict = task.tgt_dict
        self.min_top1_prob = 0.4

    def test_export_diverse_bs(self):
        if False:
            while True:
                i = 10
        search_strategy = search.DiverseBeamSearch(self.tgt_dict, num_groups=2, diversity_strength=0.0)
        torch.jit.script(search_strategy)

    def test_export_sampling(self):
        if False:
            while True:
                i = 10
        low_sampling_topp = self.min_top1_prob / 2.0
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=low_sampling_topp)
        torch.jit.script(search_strategy)

    def test_export_diverse_siblings_search(self):
        if False:
            print('Hello World!')
        search_strategy = search.DiverseSiblingsSearch(self.tgt_dict, diversity_rate=0.5)
        torch.jit.script(search_strategy)

class TestSequenceGeneratorBase(unittest.TestCase):

    def assertHypoTokens(self, hypo, tokens):
        if False:
            for i in range(10):
                print('nop')
        self.assertTensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.0):
        if False:
            while True:
                i = 10
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo['positional_scores'], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo['tokens'].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo['score']), 1e-06)

    def assertAlmostEqual(self, t1, t2):
        if False:
            i = 10
            return i + 15
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        self.assertLess((t1 - t2).abs().max(), 0.0001)

    def assertTensorEqual(self, t1, t2):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        self.assertEqual(t1.ne(t2).long().sum(), 0)

class TestSequenceGenerator(TestSequenceGeneratorBase):

    def setUp(self):
        if False:
            print('Hello World!')
        (self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model) = test_utils.sequence_generator_setup()
        self.sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}

    def test_with_normalization(self):
        if False:
            i = 10
            return i + 15
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2)
        hypos = generator.forward(self.sample)
        (eos, w1, w2) = (self.tgt_dict.eos(), self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0])
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0])
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6])

    def test_without_normalization(self):
        if False:
            while True:
                i = 10
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, normalize_scores=False)
        hypos = generator.forward(self.sample)
        (eos, w1, w2) = (self.tgt_dict.eos(), self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], normalized=False)
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], normalized=False)
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], normalized=False)
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], normalized=False)

    def test_with_lenpen_favoring_short_hypos(self):
        if False:
            print('Hello World!')
        lenpen = 0.6
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.forward(self.sample)
        (eos, w1, w2) = (self.tgt_dict.eos(), self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], lenpen=lenpen)
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], lenpen=lenpen)
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)

    def test_with_lenpen_favoring_long_hypos(self):
        if False:
            print('Hello World!')
        lenpen = 5.0
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.forward(self.sample)
        (eos, w1, w2) = (self.tgt_dict.eos(), self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][0], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        self.assertHypoTokens(hypos[0][1], [w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 1.0], lenpen=lenpen)
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6], lenpen=lenpen)

    def test_maxlen(self):
        if False:
            i = 10
            return i + 15
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.forward(self.sample)
        (eos, w1, w2) = (self.tgt_dict.eos(), self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        self.assertHypoTokens(hypos[0][1], [w2, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.1, 0.6])
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6])
        self.assertHypoTokens(hypos[1][1], [w2, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.3, 0.9, 0.01])

    def test_encoder_with_different_output_len(self):
        if False:
            i = 10
            return i + 15
        args = self.model.encoder.args
        task = test_utils.TestTranslationTask.setup_task(args, self.tgt_dict, self.tgt_dict)
        reshaping_model = test_utils.TestReshapingModel.build_model(args, task)
        generator = SequenceGenerator([reshaping_model], self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.forward(self.sample)
        for sent in [0, 1]:
            for beam in [0, 1]:
                assert hypos[sent][beam]['attention'] is not None

    def test_generation_with_additional_input(self):
        if False:
            i = 10
            return i + 15
        args = self.model.encoder.args
        task = test_utils.TestTranslationTask.setup_task(args, self.tgt_dict, self.tgt_dict)
        add_input_model = test_utils.TestAdditionalInputModel.build_model(args, task)
        generator = SequenceGenerator([add_input_model], self.tgt_dict, beam_size=2)
        sample = self.sample.copy()
        sample['net_input']['fancy_other_input'] = sample['net_input']['src_tokens']
        hypos = generator.forward(self.sample)
        (eos, w1) = (self.tgt_dict.eos(), self.w1)
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])

@unittest.skipUnless(torch.cuda.is_available(), '')
class TestRepeatNgramBlocking(TestSequenceGeneratorBase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        (cls.tgt_dict, cls.w1, cls.w2, src_tokens, src_lengths, cls.model) = test_utils.sequence_generator_setup()
        return cls

    def test_finds_repetitive_tokens(self):
        if False:
            return 10
        (bsz, vocab_size, beam_size, step) = (2, 4, 1, 3)
        generated_tok = torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.long, device='cuda')
        lprobs = torch.zeros((beam_size * bsz, vocab_size), device='cuda')
        desired_result = lprobs.new_tensor([[0.0, 0.0, -math.inf, 0.0], [0.0, 0.0, 0.0, -math.inf]])
        (cuda_ext_result, baseline_result) = self._compare_cuda_ext_to_default_implem(bsz, beam_size, generated_tok, lprobs, step, 2)
        self.assertTensorEqual(cuda_ext_result, desired_result)
        self.assertTensorEqual(baseline_result, desired_result)

    @unittest.skipIf(torch.__version__ < '1.6.0', JIT_MSG)
    def test_jit_no_extension(self):
        if False:
            i = 10
            return i + 15
        (bsz, vocab_size, beam_size, step) = (2, 4, 1, 3)
        generated_tok = torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.long, device='cuda')
        lprobs = torch.zeros((beam_size * bsz, vocab_size), device='cuda')
        blocker = NGramRepeatBlock(2, use_extension=False)
        base_result = blocker(generated_tok, lprobs.clone(), bsz, beam_size, step)
        scripted_blocker = torch.jit.script(blocker)
        jit_result = scripted_blocker(generated_tok, lprobs.clone(), bsz, beam_size, step)
        self.assertTensorEqual(base_result, jit_result)

    def test_ngram_blocking_same_as_default_implem(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that cuda extension returns same things as default impl in many settings.'
        vocab_size = 4
        step = 6
        for _ in range(2):
            block_param = np.random.choice([1, 2, 3, 4])
            batch_size = np.random.randint(1, 8)
            beam_size = np.random.choice([1, 2, 4, 8])
            lprobs = torch.zeros((beam_size * batch_size, vocab_size), device='cuda')
            generated_tok = torch.tensor(np.random.randint(0, vocab_size, size=(batch_size * beam_size, step + 1)), device='cuda', dtype=torch.long)
            self._compare_cuda_ext_to_default_implem(batch_size, beam_size, generated_tok, lprobs, step, block_param)

    def _compare_cuda_ext_to_default_implem(self, bsz, beam_size, generated_tok, lprobs, step, block_param):
        if False:
            print('Hello World!')
        'Assert that cuda extension and default implem return the same thing.'
        blocker = NGramRepeatBlock(block_param)
        assert blocker.use_extension, 'Extension not compiled'
        cuda_ext_result = blocker(generated_tok, lprobs.clone(), bsz, beam_size, step)
        blocker.use_extension = False
        baseline_result = blocker(generated_tok, lprobs.clone(), bsz, beam_size, step)
        self.assertTensorEqual(cuda_ext_result, baseline_result)
        blocker.use_extension = True
        return (cuda_ext_result, baseline_result)

class TestDiverseBeamSearch(TestSequenceGeneratorBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5
        self.src_tokens = torch.LongTensor([[self.w1, self.w2, self.eos], [self.w1, self.w2, self.eos]])
        self.src_lengths = torch.LongTensor([2, 2])
        args = argparse.Namespace()
        unk = 0.0
        args.beam_probs = [torch.FloatTensor([[0.0, unk, 0.9, 0.1], [0.0, unk, 0.9, 0.1], [0.0, unk, 0.7, 0.3], [0.0, unk, 0.7, 0.3]]), torch.FloatTensor([[0.0, unk, 0.6, 0.4], [0.0, unk, 0.6, 0.4], [0.25, unk, 0.35, 0.4], [0.25, unk, 0.35, 0.4]]), torch.FloatTensor([[1.0, unk, 0.0, 0.0], [1.0, unk, 0.0, 0.0], [0.9, unk, 0.1, 0.0], [0.9, unk, 0.1, 0.0]])]
        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_diverse_beam_search(self):
        if False:
            i = 10
            return i + 15
        search_strategy = search.DiverseBeamSearch(self.tgt_dict, num_groups=2, diversity_strength=0.0)
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.forward(sample)
        (eos, w1, w2) = (self.eos, self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0])
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.6, 1.0])
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9])
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.9])

class TestDiverseSiblingsSearch(TestDiverseBeamSearch):

    def assertHypoScore(self, hypo, pos_probs, sibling_rank, diversity_rate, normalized=True, lenpen=1.0):
        if False:
            i = 10
            return i + 15
        pos_scores = torch.FloatTensor(pos_probs).log()
        pos_scores.sub_(torch.Tensor(sibling_rank) * diversity_rate)
        self.assertAlmostEqual(hypo['positional_scores'], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo['tokens'].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo['score']), 1e-06)

    def test_diverse_beam_search(self):
        if False:
            return 10
        search_strategy = search.DiverseSiblingsSearch(self.tgt_dict, diversity_rate=0.5)
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.forward(sample)
        (eos, w1, w2) = (self.eos, self.w1, self.w2)
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0], [0, 1, 1], 0.5)
        self.assertHypoTokens(hypos[0][1], [w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.4, 1.0], [0, 2, 1], 0.5)
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9], [0, 1, 1], 0.5)
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.35, 0.9], [0, 2, 1], 0.5)

class TestTopPSamplingSearch(TestSequenceGeneratorBase):

    def setUp(self):
        if False:
            return 10
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5
        self.src_tokens = torch.LongTensor([[self.w1, self.w2, self.eos], [self.w1, self.w2, self.eos]])
        self.src_lengths = torch.LongTensor([2, 2])
        args = argparse.Namespace()
        unk = 0.0
        self.min_top2_prob = 0.75
        self.min_top1_prob = 0.4
        w1_prob = self.min_top1_prob
        w2_prob = self.min_top2_prob - self.min_top1_prob
        eos_prob = 1 - self.min_top2_prob
        args.beam_probs = [torch.FloatTensor([[0.0, unk, 1.0, 0.0], [0.0, unk, 1.0, 0.0], [0.0, unk, 1.0, 0.0], [0.0, unk, 1.0, 0.0]]), torch.FloatTensor([[eos_prob, unk, w1_prob, w2_prob], [eos_prob, unk, w1_prob, w2_prob], [eos_prob, unk, w1_prob, w2_prob], [eos_prob, unk, w1_prob, w2_prob]]), torch.FloatTensor([[1.0, unk, 0.0, 0.0], [1.0, unk, 0.0, 0.0], [1.0, unk, 0.0, 0.0], [1.0, unk, 0.0, 0.0]])]
        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_topp_sampling_search_low_prob(self):
        if False:
            print('Hello World!')
        low_sampling_topp = self.min_top1_prob / 2.0
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=low_sampling_topp)
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.forward(sample)
        (eos, w1) = (self.eos, self.w1)
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [1.0, 0.4, 1.0])
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [1.0, 0.4, 1.0])
        self.assertHypoTokens(hypos[1][0], [w1, w1, eos])
        self.assertHypoScore(hypos[1][0], [1.0, 0.4, 1.0])
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [1.0, 0.4, 1.0])

    def test_topp_sampling_search_high_prob(self):
        if False:
            i = 10
            return i + 15
        high_sampling_topp = (self.min_top1_prob + self.min_top2_prob) / 2.0
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=high_sampling_topp)
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.forward(sample)
        (eos, w1, w2) = (self.eos, self.w1, self.w2)
        self.assertTrue(self.hypoTokens(hypos[0][0], [w1, w1, eos]) or self.hypoTokens(hypos[0][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][0], [1.0, 0.4, 1.0]) or self.hypoScore(hypos[0][0], [1.0, 0.35, 1.0]))
        self.assertTrue(self.hypoTokens(hypos[0][1], [w1, w1, eos]) or self.hypoTokens(hypos[0][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][1], [1.0, 0.4, 1.0]) or self.hypoScore(hypos[0][1], [1.0, 0.35, 1.0]))
        self.assertTrue(self.hypoTokens(hypos[1][0], [w1, w1, eos]) or self.hypoTokens(hypos[1][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][0], [1.0, 0.4, 1.0]) or self.hypoScore(hypos[1][0], [1.0, 0.35, 1.0]))
        self.assertTrue(self.hypoTokens(hypos[1][1], [w1, w1, eos]) or self.hypoTokens(hypos[1][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][1], [1.0, 0.4, 1.0]) or self.hypoScore(hypos[1][1], [1.0, 0.35, 1.0]))

    def hypoTokens(self, hypo, tokens):
        if False:
            print('Hello World!')
        return self.tensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def hypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.0):
        if False:
            return 10
        pos_scores = torch.FloatTensor(pos_probs).log()
        if not self.almostEqual(hypo['positional_scores'], pos_scores):
            return False
        if pos_scores.numel() != hypo['tokens'].numel():
            return False
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        return abs(score - hypo['score']) < 1e-06

    def almostEqual(self, t1, t2):
        if False:
            i = 10
            return i + 15
        return t1.size() == t2.size() and (t1 - t2).abs().max() < 0.0001

    def tensorEqual(self, t1, t2):
        if False:
            while True:
                i = 10
        return t1.size() == t2.size() and t1.ne(t2).long().sum() == 0
if __name__ == '__main__':
    unittest.main()