import argparse
import os
import unittest
from inspect import currentframe, getframeinfo
import numpy as np
import torch
from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.dictionary import Dictionary
from fairseq.models import BaseFairseqModel, FairseqDecoder, FairseqEncoder, FairseqEncoderDecoderModel, FairseqEncoderModel, FairseqModel
from fairseq.tasks.fairseq_task import LegacyFairseqTask
DEFAULT_TEST_VOCAB_SIZE = 100

def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    if False:
        for i in range(10):
            print('nop')
    dummy_dict = Dictionary()
    for (id, _) in enumerate(range(vocab_size)):
        dummy_dict.add_symbol('{}'.format(id), 1000)
    return dummy_dict

class DummyTask(LegacyFairseqTask):

    def __init__(self, args):
        if False:
            i = 10
            return i + 15
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, 'ctc', False):
            self.dictionary.add_symbol('<ctc_blank>')
        self.tgt_dict = self.dictionary

    @property
    def target_dictionary(self):
        if False:
            print('Hello World!')
        return self.dictionary

def get_dummy_task_and_parser():
    if False:
        for i in range(10):
            print('nop')
    '\n    to build a fariseq model, we need some dummy parse and task. This function\n    is used to create dummy task and parser to faciliate model/criterion test\n\n    Note: we use FbSpeechRecognitionTask as the dummy task. You may want\n    to use other task by providing another function\n    '
    parser = argparse.ArgumentParser(description='test_dummy_s2s_task', argument_default=argparse.SUPPRESS)
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return (task, parser)

def get_dummy_input(T=100, D=80, B=5, K=100):
    if False:
        i = 10
        return i + 15
    forward_input = {}
    feature = torch.randn(B, T, D)
    src_lengths = torch.from_numpy(np.random.randint(low=1, high=T, size=B, dtype=np.int64))
    src_lengths[0] = T
    prev_output_tokens = []
    for b in range(B):
        token_length = np.random.randint(low=1, high=src_lengths[b].item() + 1)
        tokens = np.random.randint(low=0, high=K, size=token_length, dtype=np.int64)
        prev_output_tokens.append(torch.from_numpy(tokens))
    prev_output_tokens = fairseq_data_utils.collate_tokens(prev_output_tokens, pad_idx=1, eos_idx=2, left_pad=False, move_eos_to_beginning=False)
    (src_lengths, sorted_order) = src_lengths.sort(descending=True)
    forward_input['src_tokens'] = feature.index_select(0, sorted_order)
    forward_input['src_lengths'] = src_lengths
    forward_input['prev_output_tokens'] = prev_output_tokens
    return forward_input

def get_dummy_encoder_output(encoder_out_shape=(100, 80, 5)):
    if False:
        for i in range(10):
            print('nop')
    '\n    This only provides an example to generate dummy encoder output\n    '
    (T, B, D) = encoder_out_shape
    encoder_out = {}
    encoder_out['encoder_out'] = torch.from_numpy(np.random.randn(*encoder_out_shape).astype(np.float32))
    seq_lengths = torch.from_numpy(np.random.randint(low=1, high=T, size=B))
    encoder_out['encoder_padding_mask'] = torch.arange(T).view(1, T).expand(B, -1) >= seq_lengths.view(B, 1).expand(-1, T)
    encoder_out['encoder_padding_mask'].t_()
    return encoder_out

def _current_postion_info():
    if False:
        i = 10
        return i + 15
    cf = currentframe()
    frameinfo = ' (at {}:{})'.format(os.path.basename(getframeinfo(cf).filename), cf.f_back.f_lineno)
    return frameinfo

def check_encoder_output(encoder_output, batch_size=None):
    if False:
        i = 10
        return i + 15
    'we expect encoder_output to be a dict with the following\n    key/value pairs:\n    - encoder_out: a Torch.Tensor\n    - encoder_padding_mask: a binary Torch.Tensor\n    '
    if not isinstance(encoder_output, dict):
        msg = 'FairseqEncoderModel.forward(...) must be a dict' + _current_postion_info()
        return (False, msg)
    if 'encoder_out' not in encoder_output:
        msg = 'FairseqEncoderModel.forward(...) must contain encoder_out' + _current_postion_info()
        return (False, msg)
    if 'encoder_padding_mask' not in encoder_output:
        msg = 'FairseqEncoderModel.forward(...) must contain encoder_padding_mask' + _current_postion_info()
        return (False, msg)
    if not isinstance(encoder_output['encoder_out'], torch.Tensor):
        msg = 'encoder_out must be a torch.Tensor' + _current_postion_info()
        return (False, msg)
    if encoder_output['encoder_out'].dtype != torch.float32:
        msg = 'encoder_out must have float32 dtype' + _current_postion_info()
        return (False, msg)
    mask = encoder_output['encoder_padding_mask']
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            msg = 'encoder_padding_mask must be a torch.Tensor' + _current_postion_info()
            return (False, msg)
        if mask.dtype != torch.uint8 and (not hasattr(torch, 'bool') or mask.dtype != torch.bool):
            msg = 'encoder_padding_mask must have dtype of uint8' + _current_postion_info()
            return (False, msg)
        if mask.dim() != 2:
            msg = 'we expect encoder_padding_mask to be a 2-d tensor, in shape (T, B)' + _current_postion_info()
            return (False, msg)
        if batch_size is not None and mask.size(1) != batch_size:
            msg = 'we expect encoder_padding_mask to be a 2-d tensor, with size(1)' + ' being the batch size' + _current_postion_info()
            return (False, msg)
    return (True, None)

def check_decoder_output(decoder_output):
    if False:
        for i in range(10):
            print('nop')
    'we expect output from a decoder is a tuple with the following constraint:\n    - the first element is a torch.Tensor\n    - the second element can be anything (reserved for future use)\n    '
    if not isinstance(decoder_output, tuple):
        msg = 'FariseqDecoder output must be a tuple' + _current_postion_info()
        return (False, msg)
    if len(decoder_output) != 2:
        msg = 'FairseqDecoder output must be 2-elem tuple' + _current_postion_info()
        return (False, msg)
    if not isinstance(decoder_output[0], torch.Tensor):
        msg = 'FariseqDecoder output[0] must be a torch.Tensor' + _current_postion_info()
        return (False, msg)
    return (True, None)

class TestBaseFairseqModelBase(unittest.TestCase):
    """
    This class is used to facilitate writing unittest for any class derived from
    `BaseFairseqModel`.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        if cls is TestBaseFairseqModelBase:
            raise unittest.SkipTest('Skipping test case in base')
        super().setUpClass()

    def setUpModel(self, model):
        if False:
            return 10
        self.assertTrue(isinstance(model, BaseFairseqModel))
        self.model = model

    def setupInput(self):
        if False:
            while True:
                i = 10
        pass

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model = None
        self.forward_input = None
        pass

class TestFairseqEncoderDecoderModelBase(TestBaseFairseqModelBase):
    """
    base code to test FairseqEncoderDecoderModel (formally known as
    `FairseqModel`) must be derived from this base class
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        if cls is TestFairseqEncoderDecoderModelBase:
            raise unittest.SkipTest('Skipping test case in base')
        super().setUpClass()

    def setUpModel(self, model_cls, extra_args_setters=None):
        if False:
            print('Hello World!')
        self.assertTrue(issubclass(model_cls, (FairseqEncoderDecoderModel, FairseqModel)), msg='This class only tests for FairseqModel subclasses')
        (task, parser) = get_dummy_task_and_parser()
        model_cls.add_args(parser)
        args = parser.parse_args([])
        if extra_args_setters is not None:
            for args_setter in extra_args_setters:
                args_setter(args)
        model = model_cls.build_model(args, task)
        self.model = model

    def setUpInput(self, input=None):
        if False:
            print('Hello World!')
        self.forward_input = get_dummy_input() if input is None else input

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    def test_forward(self):
        if False:
            for i in range(10):
                print('nop')
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            (succ, msg) = check_decoder_output(forward_output)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output

    def test_get_normalized_probs(self):
        if False:
            return 10
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            logprob = self.model.get_normalized_probs(forward_output, log_probs=True)
            prob = self.model.get_normalized_probs(forward_output, log_probs=False)
            self.assertTrue(hasattr(logprob, 'batch_first'))
            self.assertTrue(hasattr(prob, 'batch_first'))
            self.assertTrue(torch.is_tensor(logprob))
            self.assertTrue(torch.is_tensor(prob))

class TestFairseqEncoderModelBase(TestBaseFairseqModelBase):
    """
    base class to test FairseqEncoderModel
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        if cls is TestFairseqEncoderModelBase:
            raise unittest.SkipTest('Skipping test case in base')
        super().setUpClass()

    def setUpModel(self, model_cls, extra_args_setters=None):
        if False:
            return 10
        self.assertTrue(issubclass(model_cls, FairseqEncoderModel), msg='This class is only used for testing FairseqEncoderModel')
        (task, parser) = get_dummy_task_and_parser()
        model_cls.add_args(parser)
        args = parser.parse_args([])
        if extra_args_setters is not None:
            for args_setter in extra_args_setters:
                args_setter(args)
        model = model_cls.build_model(args, task)
        self.model = model

    def setUpInput(self, input=None):
        if False:
            while True:
                i = 10
        self.forward_input = get_dummy_input() if input is None else input
        self.forward_input.pop('prev_output_tokens', None)

    def setUp(self):
        if False:
            return 10
        super().setUp()

    def test_forward(self):
        if False:
            for i in range(10):
                print('nop')
        if self.forward_input and self.model:
            bsz = self.forward_input['src_tokens'].size(0)
            forward_output = self.model.forward(**self.forward_input)
            (succ, msg) = check_encoder_output(forward_output, batch_size=bsz)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output

    def test_get_normalized_probs(self):
        if False:
            print('Hello World!')
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            logprob = self.model.get_normalized_probs(forward_output, log_probs=True)
            prob = self.model.get_normalized_probs(forward_output, log_probs=False)
            self.assertTrue(hasattr(logprob, 'batch_first'))
            self.assertTrue(hasattr(prob, 'batch_first'))
            self.assertTrue(torch.is_tensor(logprob))
            self.assertTrue(torch.is_tensor(prob))

class TestFairseqEncoderBase(unittest.TestCase):
    """
    base class to test FairseqEncoder
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        if cls is TestFairseqEncoderBase:
            raise unittest.SkipTest('Skipping test case in base')
        super().setUpClass()

    def setUpEncoder(self, encoder):
        if False:
            return 10
        self.assertTrue(isinstance(encoder, FairseqEncoder), msg='This class is only used for test FairseqEncoder')
        self.encoder = encoder

    def setUpInput(self, input=None):
        if False:
            i = 10
            return i + 15
        self.forward_input = get_dummy_input() if input is None else input
        self.forward_input.pop('prev_output_tokens', None)

    def setUp(self):
        if False:
            print('Hello World!')
        self.encoder = None
        self.forward_input = None

    def test_forward(self):
        if False:
            return 10
        if self.encoder and self.forward_input:
            bsz = self.forward_input['src_tokens'].size(0)
            forward_output = self.encoder.forward(**self.forward_input)
            (succ, msg) = check_encoder_output(forward_output, batch_size=bsz)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output

class TestFairseqDecoderBase(unittest.TestCase):
    """
    base class to test FairseqDecoder
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        if cls is TestFairseqDecoderBase:
            raise unittest.SkipTest('Skipping test case in base')
        super().setUpClass()

    def setUpDecoder(self, decoder):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(decoder, FairseqDecoder), msg='This class is only used for test FairseqDecoder')
        self.decoder = decoder

    def setUpInput(self, input=None):
        if False:
            for i in range(10):
                print('nop')
        self.forward_input = get_dummy_encoder_output() if input is None else input

    def setUpPrevOutputTokens(self, tokens=None):
        if False:
            while True:
                i = 10
        if tokens is None:
            self.encoder_input = get_dummy_input()
            self.prev_output_tokens = self.encoder_input['prev_output_tokens']
        else:
            self.prev_output_tokens = tokens

    def setUp(self):
        if False:
            print('Hello World!')
        self.decoder = None
        self.forward_input = None
        self.prev_output_tokens = None

    def test_forward(self):
        if False:
            i = 10
            return i + 15
        if self.decoder is not None and self.forward_input is not None and (self.prev_output_tokens is not None):
            forward_output = self.decoder.forward(prev_output_tokens=self.prev_output_tokens, encoder_out=self.forward_input)
            (succ, msg) = check_decoder_output(forward_output)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_input = forward_output

class DummyEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        if False:
            return 10
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):
        if False:
            for i in range(10):
                print('nop')
        return cls(DummyEncoder())

    def get_logits(self, net_output):
        if False:
            for i in range(10):
                print('nop')
        return torch.log(torch.div(net_output['encoder_out'], 1 - net_output['encoder_out']))

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            for i in range(10):
                print('nop')
        lprobs = super().get_normalized_probs(net_output, log_probs, sample=sample)
        lprobs.batch_first = True
        return lprobs

class DummyEncoder(FairseqEncoder):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(None)

    def forward(self, src_tokens, src_lengths):
        if False:
            while True:
                i = 10
        (mask, max_len) = lengths_to_encoder_padding_mask(src_lengths)
        return {'encoder_out': src_tokens, 'encoder_padding_mask': mask}

class CrossEntropyCriterionTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        if cls is CrossEntropyCriterionTestBase:
            raise unittest.SkipTest('Skipping base class test case')
        super().setUpClass()

    def setUpArgs(self):
        if False:
            while True:
                i = 10
        args = argparse.Namespace()
        args.sentence_avg = False
        args.threshold = 0.1
        return args

    def setUp(self):
        if False:
            return 10
        args = self.setUpArgs()
        self.model = DummyEncoderModel(encoder=DummyEncoder())
        self.criterion = self.criterion_cls.build_criterion(args, task=DummyTask(args))

    def get_src_tokens(self, correct_prediction, aggregate):
        if False:
            return 10
        '\n        correct_prediction: True if the net_output (src_tokens) should\n        predict the correct target\n        aggregate: True if the criterion expects net_output (src_tokens)\n        aggregated across time axis\n        '
        predicted_idx = 0 if correct_prediction else 1
        if aggregate:
            src_tokens = torch.zeros((2, 2), dtype=torch.float)
            for b in range(2):
                src_tokens[b][predicted_idx] = 1.0
        else:
            src_tokens = torch.zeros((2, 10, 2), dtype=torch.float)
            for b in range(2):
                for t in range(10):
                    src_tokens[b][t][predicted_idx] = 1.0
        return src_tokens

    def get_target(self, soft_target):
        if False:
            print('Hello World!')
        if soft_target:
            target = torch.zeros((2, 2), dtype=torch.float)
            for b in range(2):
                target[b][0] = 1.0
        else:
            target = torch.zeros((2, 10), dtype=torch.long)
        return target

    def get_test_sample(self, correct, soft_target, aggregate):
        if False:
            return 10
        src_tokens = self.get_src_tokens(correct, aggregate)
        target = self.get_target(soft_target)
        L = src_tokens.size(1)
        return {'net_input': {'src_tokens': src_tokens, 'src_lengths': torch.tensor([L])}, 'target': target, 'ntokens': src_tokens.size(0) * src_tokens.size(1)}