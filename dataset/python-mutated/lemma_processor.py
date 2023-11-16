"""
Processor for performing lemmatization
"""
from itertools import compress
import torch
from stanza.models.common import doc
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
WORD_TAGS = [doc.TEXT, doc.UPOS]

@register_processor(name=LEMMA)
class LemmaProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([LEMMA])
    REQUIRES_DEFAULT = set([TOKENIZE])
    DEFAULT_BATCH_SIZE = 5000

    def __init__(self, config, pipeline, device):
        if False:
            i = 10
            return i + 15
        self._use_identity = None
        self._pretagged = None
        super().__init__(config, pipeline, device)

    @property
    def use_identity(self):
        if False:
            for i in range(10):
                print('nop')
        return self._use_identity

    def _set_up_model(self, config, pipeline, device):
        if False:
            i = 10
            return i + 15
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self._config = config
            self.config['batch_size'] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            self.store_results = config.get('store_results', False)
            self._use_identity = False
            args = {'charlm_forward_file': config.get('forward_charlm_path', None), 'charlm_backward_file': config.get('backward_charlm_path', None)}
            self._trainer = Trainer(args=args, model_file=config['model_path'], device=device, foundation_cache=pipeline.foundation_cache)

    def _set_up_requires(self):
        if False:
            i = 10
            return i + 15
        self._pretagged = self._config.get('pretagged', None)
        if self._pretagged:
            self._requires = set()
        elif self.config.get('pos') and (not self.use_identity):
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, document):
        if False:
            for i in range(10):
                print('nop')
        if not self.use_identity:
            batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        else:
            batch = DataLoader(document, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.doc.get([doc.TEXT, doc.UPOS]))
        else:
            if self.config.get('ensemble_dict', False):
                skip = self.trainer.skip_seq2seq(batch.doc.get([doc.TEXT, doc.UPOS]))
                seq2seq_batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, skip=skip)
            else:
                seq2seq_batch = batch
            with torch.no_grad():
                preds = []
                edits = []
                for (i, b) in enumerate(seq2seq_batch):
                    (ps, es) = self.trainer.predict(b, self.config['beam_size'])
                    preds += ps
                    if es is not None:
                        edits += es
            if self.config.get('ensemble_dict', False):
                word_tags = batch.doc.get(WORD_TAGS)
                words = [x[0] for x in word_tags]
                preds = self.trainer.postprocess([x for (x, y) in zip(words, skip) if not y], preds, edits=edits)
                if self.store_results:
                    new_word_tags = compress(word_tags, map(lambda x: not x, skip))
                    new_predictions = [(x[0], x[1], y) for (x, y) in zip(new_word_tags, preds)]
                    self.trainer.train_dict(new_predictions, update_word_dict=False)
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append('')
                    else:
                        preds1.append(preds[i])
                        i += 1
                preds = self.trainer.ensemble(word_tags, preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc