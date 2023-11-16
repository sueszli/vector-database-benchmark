"""
Processor that attaches a constituency tree to a sentence
"""
from stanza.models.constituency.trainer import Trainer
from stanza.models.common import doc
from stanza.utils.get_tqdm import get_tqdm
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
tqdm = get_tqdm()

@register_processor(CONSTITUENCY)
class ConstituencyProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([CONSTITUENCY])
    REQUIRES_DEFAULT = set([TOKENIZE, POS])
    DEFAULT_BATCH_SIZE = 50

    def _set_up_requires(self):
        if False:
            return 10
        self._pretagged = self._config.get('pretagged')
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, pipeline, device):
        if False:
            i = 10
            return i + 15
        args = {'wordvec_pretrain_file': config.get('pretrain_path', None), 'charlm_forward_file': config.get('forward_charlm_path', None), 'charlm_backward_file': config.get('backward_charlm_path', None), 'device': device}
        trainer = Trainer.load(filename=config['model_path'], args=args, foundation_cache=pipeline.foundation_cache)
        self._model = trainer.model
        self._model.eval()
        self._batch_size = int(config.get('batch_size', ConstituencyProcessor.DEFAULT_BATCH_SIZE))
        self._tqdm = 'tqdm' in config and config['tqdm']

    def process(self, document):
        if False:
            while True:
                i = 10
        sentences = document.sentences
        if self._model.uses_xpos():
            words = [[(w.text, w.xpos) for w in s.words] for s in sentences]
        else:
            words = [[(w.text, w.upos) for w in s.words] for s in sentences]
        if self._tqdm:
            words = tqdm(words)
        trees = self._model.parse_tagged_words(words, self._batch_size)
        document.set(CONSTITUENCY, trees, to_sentence=True)
        return document

    def get_constituents(self):
        if False:
            while True:
                i = 10
        '\n        Return a set of the constituents known by this model\n\n        For a pipeline, this can be queried with\n          pipeline.processors["constituency"].get_constituents()\n        '
        return set(self._model.constituents)