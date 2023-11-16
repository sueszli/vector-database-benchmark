from typing import List, Dict, Any
import spacy
from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model

class SentenceSplitter(Registrable):
    """
    A `SentenceSplitter` splits strings into sentences.
    """
    default_implementation = 'spacy'

    def split_sentences(self, text: str) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Splits a `text` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.\n        '
        raise NotImplementedError

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Default implementation is to just iterate over the texts and call `split_sentences`.\n        '
        return [self.split_sentences(text) for text in texts]

@SentenceSplitter.register('spacy')
class SpacySentenceSplitter(SentenceSplitter):
    """
    A `SentenceSplitter` that uses spaCy's built-in sentence boundary detection.

    Spacy's default sentence splitter uses a dependency parse to detect sentence boundaries, so
    it is slow, but accurate.

    Another option is to use rule-based sentence boundary detection. It's fast and has a small memory footprint,
    since it uses punctuation to detect sentence boundaries. This can be activated with the `rule_based` flag.

    By default, `SpacySentenceSplitter` calls the default spacy boundary detector.

    Registered as a `SentenceSplitter` with name "spacy".
    """

    def __init__(self, language: str='en_core_web_sm', rule_based: bool=False) -> None:
        if False:
            while True:
                i = 10
        self._language = language
        self._rule_based = rule_based
        self.spacy = get_spacy_model(self._language, parse=not self._rule_based, ner=False)
        self._is_version_3 = spacy.__version__ >= '3.0'
        if rule_based:
            sbd_name = 'sbd' if spacy.__version__ < '2.1' else 'sentencizer'
            if not self.spacy.has_pipe(sbd_name):
                if self._is_version_3:
                    self.spacy.add_pipe(sbd_name)
                else:
                    sbd = self.spacy.create_pipe(sbd_name)
                    self.spacy.add_pipe(sbd)

    def split_sentences(self, text: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        if self._is_version_3:
            return [sent.text.strip() for sent in self.spacy(text).sents]
        else:
            return [sent.string.strip() for sent in self.spacy(text).sents]

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        if False:
            while True:
                i = 10
        "\n        This method lets you take advantage of spacy's batch processing.\n        "
        if self._is_version_3:
            return [[sentence.text.strip() for sentence in doc.sents] for doc in self.spacy.pipe(texts)]
        return [[sentence.string.strip() for sentence in doc.sents] for doc in self.spacy.pipe(texts)]

    def _to_params(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'type': 'spacy', 'language': self._language, 'rule_based': self._rule_based}