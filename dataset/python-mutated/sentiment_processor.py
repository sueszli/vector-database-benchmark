"""Processor that attaches a sentiment score to a sentence

The model used is a generally a model trained on the Stanford
Sentiment Treebank or some similar dataset.  When run, this processor
attaches a score in the form of a string to each sentence in the
document.

TODO: a possible way to generalize this would be to make it a
ClassifierProcessor and have "sentiment" be an option.
"""
import torch
from types import SimpleNamespace
from stanza.models.classifiers.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

@register_processor(SENTIMENT)
class SentimentProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([SENTIMENT])
    REQUIRES_DEFAULT = set([TOKENIZE])
    DEFAULT_BATCH_SIZE = 5000

    def _set_up_model(self, config, pipeline, device):
        if False:
            while True:
                i = 10
        pretrain_path = config.get('pretrain_path', None)
        forward_charlm_path = config.get('forward_charlm_path', None)
        backward_charlm_path = config.get('backward_charlm_path', None)
        args = SimpleNamespace(device=device, charlm_forward_file=forward_charlm_path, charlm_backward_file=backward_charlm_path, wordvec_pretrain_file=pretrain_path, elmo_model=None, use_elmo=False, save_dir=None)
        filename = config['model_path']
        if filename is None:
            raise FileNotFoundError('No model specified for the sentiment processor.  Perhaps it is not supported for the language.  {}'.format(config))
        trainer = Trainer.load(filename=filename, args=args, foundation_cache=pipeline.foundation_cache)
        self._model = trainer.model
        self._model_type = self._model.config.model_type
        self._batch_size = config.get('batch_size', SentimentProcessor.DEFAULT_BATCH_SIZE)

    def process(self, document):
        if False:
            i = 10
            return i + 15
        sentences = self._model.extract_sentences(document)
        with torch.no_grad():
            labels = self._model.label_sentences(sentences, batch_size=self._batch_size)
        document.set(SENTIMENT, labels, to_sentence=True)
        return document