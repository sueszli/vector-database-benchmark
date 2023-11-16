""" XNLI utils (dataset loading and evaluation)"""
import os
from ...utils import logging
from .utils import DataProcessor, InputExample
logger = logging.get_logger(__name__)

class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """

    def __init__(self, language, train_language=None):
        if False:
            print('Hello World!')
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        if False:
            return 10
        'See base class.'
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, f'XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f'train-{i}'
            text_a = line[0]
            text_b = line[1]
            label = 'contradiction' if line[2] == 'contradictory' else line[2]
            if not isinstance(text_a, str):
                raise ValueError(f'Training input {text_a} is not a string')
            if not isinstance(text_b, str):
                raise ValueError(f'Training input {text_b} is not a string')
            if not isinstance(label, str):
                raise ValueError(f'Training label {label} is not a string')
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        lines = self._read_tsv(os.path.join(data_dir, 'XNLI-1.0/xnli.test.tsv'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = f'test-{i}'
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            if not isinstance(text_a, str):
                raise ValueError(f'Training input {text_a} is not a string')
            if not isinstance(text_b, str):
                raise ValueError(f'Training input {text_b} is not a string')
            if not isinstance(label, str):
                raise ValueError(f'Training label {label} is not a string')
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        if False:
            while True:
                i = 10
        'See base class.'
        return ['contradiction', 'entailment', 'neutral']
xnli_processors = {'xnli': XnliProcessor}
xnli_output_modes = {'xnli': 'classification'}
xnli_tasks_num_labels = {'xnli': 3}