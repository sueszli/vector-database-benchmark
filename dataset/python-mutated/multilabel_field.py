from typing import Dict, Union, Sequence, Set, Optional, cast
import logging
import torch
from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError
logger = logging.getLogger(__name__)

class MultiLabelField(Field[torch.Tensor]):
    """
    A `MultiLabelField` is an extension of the :class:`LabelField` that allows for multiple labels.
    It is particularly useful in multi-label classification where more than one label can be correct.
    As with the :class:`LabelField`, labels are either strings of text or 0-indexed integers (if you wish
    to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.

    This field will get converted into a vector of length equal to the vocabulary size with
    one hot encoding for the labels (all zeros, and ones for the labels).

    # Parameters

    labels : `Sequence[Union[str, int]]`
    label_namespace : `str`, optional (default=`"labels"`)
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the `Vocabulary` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : `bool`, optional (default=`False`)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is `False` and your labels are not strings, this throws a `ConfigurationError`.
    num_labels : `int`, optional (default=`None`)
        If `skip_indexing=True`, the total number of possible labels should be provided, which is required
        to decide the size of the output tensor. `num_labels` should equal largest label id + 1.
        If `skip_indexing=False`, `num_labels` is not required.

    """
    __slots__ = ['labels', '_label_namespace', '_label_ids', '_num_labels']
    _already_warned_namespaces: Set[str] = set()

    def __init__(self, labels: Sequence[Union[str, int]], label_namespace: str='labels', skip_indexing: bool=False, num_labels: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.labels = labels
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels = num_labels
        if skip_indexing and self.labels:
            if not all((isinstance(label, int) for label in labels)):
                raise ConfigurationError('In order to skip indexing, your labels must be integers. Found labels = {}'.format(labels))
            if not num_labels:
                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")
            if not all((cast(int, label) < num_labels for label in labels)):
                raise ConfigurationError('All labels should be < num_labels. Found num_labels = {} and labels = {} '.format(num_labels, labels))
            self._label_ids = labels
        elif not all((isinstance(label, str) for label in labels)):
            raise ConfigurationError('MultiLabelFields expects string labels if skip_indexing=False. Found labels: {}'.format(labels))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not (label_namespace.endswith('labels') or label_namespace.endswith('tags')):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.", self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if False:
            while True:
                i = 10
        if self._label_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1

    def index(self, vocab: Vocabulary):
        if False:
            return 10
        if self._label_ids is None:
            self._label_ids = [vocab.get_token_index(label, self._label_namespace) for label in self.labels]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        if False:
            return 10
        tensor = torch.zeros(self._num_labels, dtype=torch.long)
        if self._label_ids:
            tensor.scatter_(0, torch.LongTensor(self._label_ids), 1)
        return tensor

    def empty_field(self):
        if False:
            for i in range(10):
                print('nop')
        return MultiLabelField([], self._label_namespace, skip_indexing=True, num_labels=self._num_labels)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f"MultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 1

    def human_readable_repr(self) -> Sequence[Union[str, int]]:
        if False:
            return 10
        return self.labels