from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
spancat_default_config = '\n[model]\n@architectures = "spacy.SpanCategorizer.v1"\nscorer = {"@layers": "spacy.LinearLogistic.v1"}\n\n[model.reducer]\n@layers = spacy.mean_max_reducer.v1\nhidden_size = 128\n\n[model.tok2vec]\n@architectures = "spacy.Tok2Vec.v2"\n\n[model.tok2vec.embed]\n@architectures = "spacy.MultiHashEmbed.v2"\nwidth = 96\nrows = [5000, 1000, 2500, 1000]\nattrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]\ninclude_static_vectors = false\n\n[model.tok2vec.encode]\n@architectures = "spacy.MaxoutWindowEncoder.v2"\nwidth = ${model.tok2vec.embed.width}\nwindow_size = 1\nmaxout_pieces = 3\ndepth = 4\n'
spancat_singlelabel_default_config = '\n[model]\n@architectures = "spacy.SpanCategorizer.v1"\nscorer = {"@layers": "Softmax.v2"}\n\n[model.reducer]\n@layers = spacy.mean_max_reducer.v1\nhidden_size = 128\n\n[model.tok2vec]\n@architectures = "spacy.Tok2Vec.v2"\n[model.tok2vec.embed]\n@architectures = "spacy.MultiHashEmbed.v1"\nwidth = 96\nrows = [5000, 1000, 2500, 1000]\nattrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]\ninclude_static_vectors = false\n\n[model.tok2vec.encode]\n@architectures = "spacy.MaxoutWindowEncoder.v2"\nwidth = ${model.tok2vec.embed.width}\nwindow_size = 1\nmaxout_pieces = 3\ndepth = 4\n'
DEFAULT_SPANS_KEY = 'sc'
DEFAULT_SPANCAT_MODEL = Config().from_str(spancat_default_config)['model']
DEFAULT_SPANCAT_SINGLELABEL_MODEL = Config().from_str(spancat_singlelabel_default_config)['model']

@runtime_checkable
class Suggester(Protocol):

    def __call__(self, docs: Iterable[Doc], *, ops: Optional[Ops]=None) -> Ragged:
        if False:
            while True:
                i = 10
        ...

def ngram_suggester(docs: Iterable[Doc], sizes: List[int], *, ops: Optional[Ops]=None) -> Ragged:
    if False:
        print('Hello World!')
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []
    for doc in docs:
        starts = ops.xp.arange(len(doc), dtype='i')
        starts = starts.reshape((-1, 1))
        length = 0
        for size in sizes:
            if size <= len(doc):
                starts_size = starts[:len(doc) - (size - 1)]
                spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                length += spans[-1].shape[0]
            if spans:
                assert spans[-1].ndim == 2, spans[-1].shape
        lengths.append(length)
    lengths_array = ops.asarray1i(lengths)
    if len(spans) > 0:
        output = Ragged(ops.xp.vstack(spans), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype='i'), lengths_array)
    assert output.dataXd.ndim == 2
    return output

def preset_spans_suggester(docs: Iterable[Doc], spans_key: str, *, ops: Optional[Ops]=None) -> Ragged:
    if False:
        return 10
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []
    for doc in docs:
        length = 0
        if doc.spans[spans_key]:
            for span in doc.spans[spans_key]:
                spans.append([span.start, span.end])
                length += 1
        lengths.append(length)
    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype='i'))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype='i'), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype='i'), lengths_array)
    return output

@registry.misc('spacy.ngram_suggester.v1')
def build_ngram_suggester(sizes: List[int]) -> Suggester:
    if False:
        return 10
    'Suggest all spans of the given lengths. Spans are returned as a ragged\n    array of integers. The array has two columns, indicating the start and end\n    position.'
    return partial(ngram_suggester, sizes=sizes)

@registry.misc('spacy.ngram_range_suggester.v1')
def build_ngram_range_suggester(min_size: int, max_size: int) -> Suggester:
    if False:
        return 10
    'Suggest all spans of the given lengths between a given min and max value - both inclusive.\n    Spans are returned as a ragged array of integers. The array has two columns,\n    indicating the start and end position.'
    sizes = list(range(min_size, max_size + 1))
    return build_ngram_suggester(sizes)

@registry.misc('spacy.preset_spans_suggester.v1')
def build_preset_spans_suggester(spans_key: str) -> Suggester:
    if False:
        print('Hello World!')
    'Suggest all spans that are already stored in doc.spans[spans_key].\n    This is useful when an upstream component is used to set the spans\n    on the Doc such as a SpanRuler or SpanFinder.'
    return partial(preset_spans_suggester, spans_key=spans_key)

@Language.factory('spancat', assigns=['doc.spans'], default_config={'threshold': 0.5, 'spans_key': DEFAULT_SPANS_KEY, 'max_positive': None, 'model': DEFAULT_SPANCAT_MODEL, 'suggester': {'@misc': 'spacy.ngram_suggester.v1', 'sizes': [1, 2, 3]}, 'scorer': {'@scorers': 'spacy.spancat_scorer.v1'}}, default_score_weights={'spans_sc_f': 1.0, 'spans_sc_p': 0.0, 'spans_sc_r': 0.0})
def make_spancat(nlp: Language, name: str, suggester: Suggester, model: Model[Tuple[List[Doc], Ragged], Floats2d], spans_key: str, scorer: Optional[Callable], threshold: float, max_positive: Optional[int]) -> 'SpanCategorizer':
    if False:
        for i in range(10):
            print('nop')
    'Create a SpanCategorizer component and configure it for multi-label\n    classification to be able to assign multiple labels for each span.\n    The span categorizer consists of two\n    parts: a suggester function that proposes candidate spans, and a labeller\n    model that predicts one or more labels for each span.\n\n    name (str): The component instance name, used to add entries to the\n        losses during training.\n    suggester (Callable[[Iterable[Doc], Optional[Ops]], Ragged]): A function that suggests spans.\n        Spans are returned as a ragged array with two integer columns, for the\n        start and end positions.\n    model (Model[Tuple[List[Doc], Ragged], Floats2d]): A model instance that\n        is given a list of documents and (start, end) indices representing\n        candidate span offsets. The model predicts a probability for each category\n        for each span.\n    spans_key (str): Key of the doc.spans dict to save the spans under. During\n        initialization and training, the component will look for spans on the\n        reference document under the same key.\n    scorer (Optional[Callable]): The scoring method. Defaults to\n        Scorer.score_spans for the Doc.spans[spans_key] with overlapping\n        spans allowed.\n    threshold (float): Minimum probability to consider a prediction positive.\n        Spans with a positive prediction will be saved on the Doc. Defaults to\n        0.5.\n    max_positive (Optional[int]): Maximum number of labels to consider positive\n        per span. Defaults to None, indicating no limit.\n    '
    return SpanCategorizer(nlp.vocab, model=model, suggester=suggester, name=name, spans_key=spans_key, negative_weight=None, allow_overlap=True, max_positive=max_positive, threshold=threshold, scorer=scorer, add_negative_label=False)

@Language.factory('spancat_singlelabel', assigns=['doc.spans'], default_config={'spans_key': DEFAULT_SPANS_KEY, 'model': DEFAULT_SPANCAT_SINGLELABEL_MODEL, 'negative_weight': 1.0, 'suggester': {'@misc': 'spacy.ngram_suggester.v1', 'sizes': [1, 2, 3]}, 'scorer': {'@scorers': 'spacy.spancat_scorer.v1'}, 'allow_overlap': True}, default_score_weights={'spans_sc_f': 1.0, 'spans_sc_p': 0.0, 'spans_sc_r': 0.0})
def make_spancat_singlelabel(nlp: Language, name: str, suggester: Suggester, model: Model[Tuple[List[Doc], Ragged], Floats2d], spans_key: str, negative_weight: float, allow_overlap: bool, scorer: Optional[Callable]) -> 'SpanCategorizer':
    if False:
        for i in range(10):
            print('nop')
    'Create a SpanCategorizer component and configure it for multi-class\n    classification. With this configuration each span can get at most one\n    label. The span categorizer consists of two\n    parts: a suggester function that proposes candidate spans, and a labeller\n    model that predicts one or more labels for each span.\n\n    name (str): The component instance name, used to add entries to the\n        losses during training.\n    suggester (Callable[[Iterable[Doc], Optional[Ops]], Ragged]): A function that suggests spans.\n        Spans are returned as a ragged array with two integer columns, for the\n        start and end positions.\n    model (Model[Tuple[List[Doc], Ragged], Floats2d]): A model instance that\n        is given a list of documents and (start, end) indices representing\n        candidate span offsets. The model predicts a probability for each category\n        for each span.\n    spans_key (str): Key of the doc.spans dict to save the spans under. During\n        initialization and training, the component will look for spans on the\n        reference document under the same key.\n    scorer (Optional[Callable]): The scoring method. Defaults to\n        Scorer.score_spans for the Doc.spans[spans_key] with overlapping\n        spans allowed.\n    negative_weight (float): Multiplier for the loss terms.\n        Can be used to downweight the negative samples if there are too many.\n    allow_overlap (bool): If True the data is assumed to contain overlapping spans.\n        Otherwise it produces non-overlapping spans greedily prioritizing\n        higher assigned label scores.\n    '
    return SpanCategorizer(nlp.vocab, model=model, suggester=suggester, name=name, spans_key=spans_key, negative_weight=negative_weight, allow_overlap=allow_overlap, max_positive=1, add_negative_label=True, threshold=None, scorer=scorer)

def spancat_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    if False:
        return 10
    kwargs = dict(kwargs)
    attr_prefix = 'spans_'
    key = kwargs['spans_key']
    kwargs.setdefault('attr', f'{attr_prefix}{key}')
    kwargs.setdefault('allow_overlap', True)
    kwargs.setdefault('getter', lambda doc, key: doc.spans.get(key[len(attr_prefix):], []))
    kwargs.setdefault('has_annotation', lambda doc: key in doc.spans)
    return Scorer.score_spans(examples, **kwargs)

@registry.scorers('spacy.spancat_scorer.v1')
def make_spancat_scorer():
    if False:
        while True:
            i = 10
    return spancat_score

@dataclass
class _Intervals:
    """
    Helper class to avoid storing overlapping spans.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.ranges = set()

    def add(self, i, j):
        if False:
            return 10
        for e in range(i, j):
            self.ranges.add(e)

    def __contains__(self, rang):
        if False:
            i = 10
            return i + 15
        (i, j) = rang
        for e in range(i, j):
            if e in self.ranges:
                return True
        return False

class SpanCategorizer(TrainablePipe):
    """Pipeline component to label spans of text.

    DOCS: https://spacy.io/api/spancategorizer
    """

    def __init__(self, vocab: Vocab, model: Model[Tuple[List[Doc], Ragged], Floats2d], suggester: Suggester, name: str='spancat', *, add_negative_label: bool=False, spans_key: str='spans', negative_weight: Optional[float]=1.0, allow_overlap: Optional[bool]=True, max_positive: Optional[int]=None, threshold: Optional[float]=0.5, scorer: Optional[Callable]=spancat_score) -> None:
        if False:
            print('Hello World!')
        'Initialize the multi-label or multi-class span categorizer.\n\n        vocab (Vocab): The shared vocabulary.\n        model (thinc.api.Model): The Thinc Model powering the pipeline component.\n            For multi-class classification (single label per span) we recommend\n            using a Softmax classifier as a the final layer, while for multi-label\n            classification (multiple possible labels per span) we recommend Logistic.\n        suggester (Callable[[Iterable[Doc], Optional[Ops]], Ragged]): A function that suggests spans.\n            Spans are returned as a ragged array with two integer columns, for the\n            start and end positions.\n        name (str): The component instance name, used to add entries to the\n            losses during training.\n        spans_key (str): Key of the Doc.spans dict to save the spans under.\n            During initialization and training, the component will look for\n            spans on the reference document under the same key. Defaults to\n            `"spans"`.\n        add_negative_label (bool): Learn to predict a special \'negative_label\'\n            when a Span is not annotated.\n        threshold (Optional[float]): Minimum probability to consider a prediction\n            positive. Defaults to 0.5. Spans with a positive prediction will be saved\n            on the Doc.\n        max_positive (Optional[int]): Maximum number of labels to consider\n            positive per span. Defaults to None, indicating no limit.\n        negative_weight (float): Multiplier for the loss terms.\n            Can be used to downweight the negative samples if there are too many\n            when add_negative_label is True. Otherwise its unused.\n        allow_overlap (bool): If True the data is assumed to contain overlapping spans.\n            Otherwise it produces non-overlapping spans greedily prioritizing\n            higher assigned label scores. Only used when max_positive is 1.\n        scorer (Optional[Callable]): The scoring method. Defaults to\n            Scorer.score_spans for the Doc.spans[spans_key] with overlapping\n            spans allowed.\n\n        DOCS: https://spacy.io/api/spancategorizer#init\n        '
        self.cfg = {'labels': [], 'spans_key': spans_key, 'threshold': threshold, 'max_positive': max_positive, 'negative_weight': negative_weight, 'allow_overlap': allow_overlap}
        self.vocab = vocab
        self.suggester = suggester
        self.model = model
        self.name = name
        self.scorer = scorer
        self.add_negative_label = add_negative_label
        if not allow_overlap and max_positive is not None and (max_positive > 1):
            raise ValueError(Errors.E1051.format(max_positive=max_positive))

    @property
    def key(self) -> str:
        if False:
            print('Hello World!')
        'Key of the doc.spans dict to save the spans under. During\n        initialization and training, the component will look for spans on the\n        reference document under the same key.\n        '
        return str(self.cfg['spans_key'])

    def _allow_extra_label(self) -> None:
        if False:
            i = 10
            return i + 15
        'Raise an error if the component can not add any more labels.'
        nO = None
        if self.model.has_dim('nO'):
            nO = self.model.get_dim('nO')
        elif self.model.has_ref('output_layer') and self.model.get_ref('output_layer').has_dim('nO'):
            nO = self.model.get_ref('output_layer').get_dim('nO')
        if nO is not None and nO == self._n_labels:
            if not self.is_resizable:
                raise ValueError(Errors.E922.format(name=self.name, nO=self.model.get_dim('nO')))

    def add_label(self, label: str) -> int:
        if False:
            return 10
        'Add a new label to the pipe.\n\n        label (str): The label to add.\n        RETURNS (int): 0 if label is already present, otherwise 1.\n\n        DOCS: https://spacy.io/api/spancategorizer#add_label\n        '
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._allow_extra_label()
        self.cfg['labels'].append(label)
        self.vocab.strings.add(label)
        return 1

    @property
    def labels(self) -> Tuple[str]:
        if False:
            return 10
        'RETURNS (Tuple[str]): The labels currently added to the component.\n\n        DOCS: https://spacy.io/api/spancategorizer#labels\n        '
        return tuple(self.cfg['labels'])

    @property
    def label_data(self) -> List[str]:
        if False:
            while True:
                i = 10
        "RETURNS (List[str]): Information about the component's labels.\n\n        DOCS: https://spacy.io/api/spancategorizer#label_data\n        "
        return list(self.labels)

    @property
    def _label_map(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        'RETURNS (Dict[str, int]): The label map.'
        return {label: i for (i, label) in enumerate(self.labels)}

    @property
    def _n_labels(self) -> int:
        if False:
            while True:
                i = 10
        'RETURNS (int): Number of labels.'
        if self.add_negative_label:
            return len(self.labels) + 1
        else:
            return len(self.labels)

    @property
    def _negative_label_i(self) -> Union[int, None]:
        if False:
            while True:
                i = 10
        'RETURNS (Union[int, None]): Index of the negative label.'
        if self.add_negative_label:
            return len(self.label_data)
        else:
            return None

    def predict(self, docs: Iterable[Doc]):
        if False:
            for i in range(10):
                print('nop')
        "Apply the pipeline's model to a batch of docs, without modifying them.\n\n        docs (Iterable[Doc]): The documents to predict.\n        RETURNS: The models prediction for each document.\n\n        DOCS: https://spacy.io/api/spancategorizer#predict\n        "
        indices = self.suggester(docs, ops=self.model.ops)
        if indices.lengths.sum() == 0:
            scores = self.model.ops.alloc2f(0, 0)
        else:
            scores = self.model.predict((docs, indices))
        return (indices, scores)

    def set_candidates(self, docs: Iterable[Doc], *, candidates_key: str='candidates') -> None:
        if False:
            i = 10
            return i + 15
        'Use the spancat suggester to add a list of span candidates to a list of docs.\n        This method is intended to be used for debugging purposes.\n\n        docs (Iterable[Doc]): The documents to modify.\n        candidates_key (str): Key of the Doc.spans dict to save the candidate spans under.\n\n        DOCS: https://spacy.io/api/spancategorizer#set_candidates\n        '
        suggester_output = self.suggester(docs, ops=self.model.ops)
        for (candidates, doc) in zip(suggester_output, docs):
            doc.spans[candidates_key] = []
            for index in candidates.dataXd:
                doc.spans[candidates_key].append(doc[index[0]:index[1]])

    def set_annotations(self, docs: Iterable[Doc], indices_scores) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Modify a batch of Doc objects, using pre-computed scores.\n\n        docs (Iterable[Doc]): The documents to modify.\n        scores: The scores to set, produced by SpanCategorizer.predict.\n\n        DOCS: https://spacy.io/api/spancategorizer#set_annotations\n        '
        (indices, scores) = indices_scores
        offset = 0
        for (i, doc) in enumerate(docs):
            indices_i = indices[i].dataXd
            allow_overlap = cast(bool, self.cfg['allow_overlap'])
            if self.cfg['max_positive'] == 1:
                doc.spans[self.key] = self._make_span_group_singlelabel(doc, indices_i, scores[offset:offset + indices.lengths[i]], allow_overlap)
            else:
                doc.spans[self.key] = self._make_span_group_multilabel(doc, indices_i, scores[offset:offset + indices.lengths[i]])
            offset += indices.lengths[i]

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        if False:
            while True:
                i = 10
        "Learn from a batch of documents and gold-standard information,\n        updating the pipe's model. Delegates to predict and get_loss.\n\n        examples (Iterable[Example]): A batch of Example objects.\n        drop (float): The dropout rate.\n        sgd (thinc.api.Optimizer): The optimizer.\n        losses (Dict[str, float]): Optional record of the loss during training.\n            Updated using the component name as the key.\n        RETURNS (Dict[str, float]): The updated losses dictionary.\n\n        DOCS: https://spacy.io/api/spancategorizer#update\n        "
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, 'SpanCategorizer.update')
        self._validate_categories(examples)
        if not any((len(eg.predicted) if eg.predicted else 0 for eg in examples)):
            return losses
        docs = [eg.predicted for eg in examples]
        spans = self.suggester(docs, ops=self.model.ops)
        if spans.lengths.sum() == 0:
            return losses
        set_dropout_rate(self.model, drop)
        (scores, backprop_scores) = self.model.begin_update((docs, spans))
        (loss, d_scores) = self.get_loss(examples, (spans, scores))
        backprop_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, examples: Iterable[Example], spans_scores: Tuple[Ragged, Floats2d]) -> Tuple[float, float]:
        if False:
            i = 10
            return i + 15
        "Find the loss and gradient of loss for the batch of documents and\n        their predicted scores.\n\n        examples (Iterable[Examples]): The batch of examples.\n        spans_scores: Scores representing the model's predictions.\n        RETURNS (Tuple[float, float]): The loss and the gradient.\n\n        DOCS: https://spacy.io/api/spancategorizer#get_loss\n        "
        (spans, scores) = spans_scores
        spans = Ragged(self.model.ops.to_numpy(spans.data), self.model.ops.to_numpy(spans.lengths))
        target = numpy.zeros(scores.shape, dtype=scores.dtype)
        if self.add_negative_label:
            negative_spans = numpy.ones(scores.shape[0])
        offset = 0
        label_map = self._label_map
        for (i, eg) in enumerate(examples):
            spans_index = {}
            spans_i = spans[i].dataXd
            for j in range(spans.lengths[i]):
                start = int(spans_i[j, 0])
                end = int(spans_i[j, 1])
                spans_index[start, end] = offset + j
            for gold_span in self._get_aligned_spans(eg):
                key = (gold_span.start, gold_span.end)
                if key in spans_index:
                    row = spans_index[key]
                    k = label_map[gold_span.label_]
                    target[row, k] = 1.0
                    if self.add_negative_label:
                        negative_spans[row] = 0.0
            offset += spans.lengths[i]
        target = self.model.ops.asarray(target, dtype='f')
        if self.add_negative_label:
            negative_samples = numpy.nonzero(negative_spans)[0]
            target[negative_samples, self._negative_label_i] = 1.0
        d_scores = scores - target
        if self.add_negative_label:
            neg_weight = cast(float, self.cfg['negative_weight'])
            if neg_weight != 1.0:
                d_scores[negative_samples] *= neg_weight
        loss = float((d_scores ** 2).sum())
        return (loss, d_scores)

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None, labels: Optional[List[str]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the pipe for training, using a representative set\n        of data examples.\n\n        get_examples (Callable[[], Iterable[Example]]): Function that\n            returns a representative sample of gold-standard Example objects.\n        nlp (Optional[Language]): The current nlp object the component is part of.\n        labels (Optional[List[str]]): The labels to add to the component, typically generated by the\n            `init labels` command. If no labels are provided, the get_examples\n            callback is used to extract the labels from the data.\n\n        DOCS: https://spacy.io/api/spancategorizer#initialize\n        '
        subbatch: List[Example] = []
        if labels is not None:
            for label in labels:
                self.add_label(label)
        for eg in get_examples():
            if labels is None:
                for span in eg.reference.spans.get(self.key, []):
                    self.add_label(span.label_)
            if len(subbatch) < 10:
                subbatch.append(eg)
        self._require_labels()
        if subbatch:
            docs = [eg.x for eg in subbatch]
            spans = build_ngram_suggester(sizes=[1])(docs)
            Y = self.model.ops.alloc2f(spans.dataXd.shape[0], self._n_labels)
            self.model.initialize(X=(docs, spans), Y=Y)
        else:
            self.model.initialize()

    def _validate_categories(self, examples: Iterable[Example]):
        if False:
            while True:
                i = 10
        pass

    def _get_aligned_spans(self, eg: Example):
        if False:
            while True:
                i = 10
        return eg.get_aligned_spans_y2x(eg.reference.spans.get(self.key, []), allow_overlap=True)

    def _make_span_group_multilabel(self, doc: Doc, indices: Ints2d, scores: Floats2d) -> SpanGroup:
        if False:
            i = 10
            return i + 15
        'Find the top-k labels for each span (k=max_positive).'
        spans = SpanGroup(doc, name=self.key)
        if scores.size == 0:
            return spans
        scores = self.model.ops.to_numpy(scores)
        indices = self.model.ops.to_numpy(indices)
        threshold = self.cfg['threshold']
        max_positive = self.cfg['max_positive']
        keeps = scores >= threshold
        if max_positive is not None:
            assert isinstance(max_positive, int)
            if self.add_negative_label:
                negative_scores = numpy.copy(scores[:, self._negative_label_i])
                scores[:, self._negative_label_i] = -numpy.inf
                ranked = (scores * -1).argsort()
                scores[:, self._negative_label_i] = negative_scores
            else:
                ranked = (scores * -1).argsort()
            span_filter = ranked[:, max_positive:]
            for (i, row) in enumerate(span_filter):
                keeps[i, row] = False
        attrs_scores = []
        for i in range(indices.shape[0]):
            start = indices[i, 0]
            end = indices[i, 1]
            for (j, keep) in enumerate(keeps[i]):
                if keep:
                    if j != self._negative_label_i:
                        spans.append(Span(doc, start, end, label=self.labels[j]))
                        attrs_scores.append(scores[i, j])
        spans.attrs['scores'] = numpy.array(attrs_scores)
        return spans

    def _make_span_group_singlelabel(self, doc: Doc, indices: Ints2d, scores: Floats2d, allow_overlap: bool=True) -> SpanGroup:
        if False:
            return 10
        'Find the argmax label for each span.'
        if scores.size == 0:
            return SpanGroup(doc, name=self.key)
        scores = self.model.ops.to_numpy(scores)
        indices = self.model.ops.to_numpy(indices)
        predicted = scores.argmax(axis=1)
        argmax_scores = numpy.take_along_axis(scores, numpy.expand_dims(predicted, 1), axis=1)
        keeps = numpy.ones(predicted.shape, dtype=bool)
        if self.add_negative_label:
            keeps = numpy.logical_and(keeps, predicted != self._negative_label_i)
        threshold = self.cfg['threshold']
        if threshold is not None:
            keeps = numpy.logical_and(keeps, (argmax_scores >= threshold).squeeze())
        if not allow_overlap:
            sort_idx = (argmax_scores.squeeze() * -1).argsort()
            argmax_scores = argmax_scores[sort_idx]
            predicted = predicted[sort_idx]
            indices = indices[sort_idx]
            keeps = keeps[sort_idx]
        seen = _Intervals()
        spans = SpanGroup(doc, name=self.key)
        attrs_scores = []
        for i in range(indices.shape[0]):
            if not keeps[i]:
                continue
            label = predicted[i]
            start = indices[i, 0]
            end = indices[i, 1]
            if not allow_overlap:
                if (start, end) in seen:
                    continue
                else:
                    seen.add(start, end)
            attrs_scores.append(argmax_scores[i])
            spans.append(Span(doc, start, end, label=self.labels[label]))
        spans.attrs['scores'] = numpy.array(attrs_scores)
        return spans