import logging
from typing import Dict, List, Set
from overrides import overrides
from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
logger = logging.getLogger(__name__)

@TokenIndexer.register('dependency_label')
class DepLabelIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens by their syntactic dependency label, as determined
    by the ``dep_`` field on ``Token``.

    Parameters
    ----------
    namespace : ``str``, optional (default=``dep_labels``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    """

    def __init__(self, namespace: str='dep_labels') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.namespace = namespace
        self._logged_errors: Set[str] = set()

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if False:
            for i in range(10):
                print('nop')
        dep_label = token.dep_
        if not dep_label:
            if token.text not in self._logged_errors:
                logger.warning('Token had no dependency label: %s', token.text)
                self._logged_errors.add(token.text)
            dep_label = 'NONE'
        counter[self.namespace][dep_label] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> int:
        if False:
            while True:
                i = 10
        dep_label = token.dep_ or 'NONE'
        return vocabulary.get_token_index(dep_label, self.namespace)

    @overrides
    def get_padding_token(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        if False:
            print('Hello World!')
        return {}

    @overrides
    def pad_token_sequence(self, tokens: List[int], desired_num_tokens: int, padding_lengths: Dict[str, int]) -> List[int]:
        if False:
            print('Hello World!')
        return pad_sequence_to_length(tokens, desired_num_tokens)

    @classmethod
    def from_params(cls, params: Params) -> 'DepLabelIndexer':
        if False:
            while True:
                i = 10
        namespace = params.pop('namespace', 'dep_labels')
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace)