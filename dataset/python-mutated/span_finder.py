from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import registry
from .spancat import DEFAULT_SPANS_KEY
from .trainable_pipe import TrainablePipe
span_finder_default_config = '\n[model]\n@architectures = "spacy.SpanFinder.v1"\n\n[model.scorer]\n@layers = "spacy.LinearLogistic.v1"\nnO = 2\n\n[model.tok2vec]\n@architectures = "spacy.Tok2Vec.v2"\n\n[model.tok2vec.embed]\n@architectures = "spacy.MultiHashEmbed.v2"\nwidth = 96\nrows = [5000, 1000, 2500, 1000]\nattrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]\ninclude_static_vectors = false\n\n[model.tok2vec.encode]\n@architectures = "spacy.MaxoutWindowEncoder.v2"\nwidth = ${model.tok2vec.embed.width}\nwindow_size = 1\nmaxout_pieces = 3\ndepth = 4\n'
DEFAULT_SPAN_FINDER_MODEL = Config().from_str(span_finder_default_config)['model']

@Language.factory('span_finder', assigns=['doc.spans'], default_config={'threshold': 0.5, 'model': DEFAULT_SPAN_FINDER_MODEL, 'spans_key': DEFAULT_SPANS_KEY, 'max_length': 25, 'min_length': None, 'scorer': {'@scorers': 'spacy.span_finder_scorer.v1'}}, default_score_weights={f'spans_{DEFAULT_SPANS_KEY}_f': 1.0, f'spans_{DEFAULT_SPANS_KEY}_p': 0.0, f'spans_{DEFAULT_SPANS_KEY}_r': 0.0})
def make_span_finder(nlp: Language, name: str, model: Model[Iterable[Doc], Floats2d], spans_key: str, threshold: float, max_length: Optional[int], min_length: Optional[int], scorer: Optional[Callable]) -> 'SpanFinder':
    if False:
        while True:
            i = 10
    'Create a SpanFinder component. The component predicts whether a token is\n    the start or the end of a potential span.\n\n    model (Model[List[Doc], Floats2d]): A model instance that\n        is given a list of documents and predicts a probability for each token.\n    spans_key (str): Key of the doc.spans dict to save the spans under. During\n        initialization and training, the component will look for spans on the\n        reference document under the same key.\n    threshold (float): Minimum probability to consider a prediction positive.\n    max_length (Optional[int]): Maximum length of the produced spans, defaults\n        to None meaning unlimited length.\n    min_length (Optional[int]): Minimum length of the produced spans, defaults\n        to None meaning shortest span length is 1.\n    scorer (Optional[Callable]): The scoring method. Defaults to\n        Scorer.score_spans for the Doc.spans[spans_key] with overlapping\n        spans allowed.\n    '
    return SpanFinder(nlp, model=model, threshold=threshold, name=name, scorer=scorer, max_length=max_length, min_length=min_length, spans_key=spans_key)

@registry.scorers('spacy.span_finder_scorer.v1')
def make_span_finder_scorer():
    if False:
        print('Hello World!')
    return span_finder_score

def span_finder_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    kwargs = dict(kwargs)
    attr_prefix = 'spans_'
    key = kwargs['spans_key']
    kwargs.setdefault('attr', f'{attr_prefix}{key}')
    kwargs.setdefault('getter', lambda doc, key: doc.spans.get(key[len(attr_prefix):], []))
    kwargs.setdefault('has_annotation', lambda doc: key in doc.spans)
    kwargs.setdefault('allow_overlap', True)
    kwargs.setdefault('labeled', False)
    scores = Scorer.score_spans(examples, **kwargs)
    scores.pop(f"{kwargs['attr']}_per_type", None)
    return scores

def _char_indices(span: Span) -> Tuple[int, int]:
    if False:
        return 10
    start = span[0].idx
    end = span[-1].idx + len(span[-1])
    return (start, end)

class SpanFinder(TrainablePipe):
    """Pipeline that learns span boundaries.

    DOCS: https://spacy.io/api/spanfinder
    """

    def __init__(self, nlp: Language, model: Model[Iterable[Doc], Floats2d], name: str='span_finder', *, spans_key: str=DEFAULT_SPANS_KEY, threshold: float=0.5, max_length: Optional[int]=None, min_length: Optional[int]=None, scorer: Optional[Callable]=span_finder_score) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the span finder.\n        model (thinc.api.Model): The Thinc Model powering the pipeline\n            component.\n        name (str): The component instance name, used to add entries to the\n            losses during training.\n        threshold (float): Minimum probability to consider a prediction\n            positive.\n        scorer (Optional[Callable]): The scoring method.\n        spans_key (str): Key of the doc.spans dict to save the spans under.\n            During initialization and training, the component will look for\n            spans on the reference document under the same key.\n        max_length (Optional[int]): Maximum length of the produced spans,\n            defaults to None meaning unlimited length.\n        min_length (Optional[int]): Minimum length of the produced spans,\n            defaults to None meaning shortest span length is 1.\n\n        DOCS: https://spacy.io/api/spanfinder#init\n        '
        self.vocab = nlp.vocab
        if max_length is not None and max_length < 1 or (min_length is not None and min_length < 1):
            raise ValueError(Errors.E1053.format(min_length=min_length, max_length=max_length))
        self.model = model
        self.name = name
        self.scorer = scorer
        self.cfg: Dict[str, Any] = {'min_length': min_length, 'max_length': max_length, 'threshold': threshold, 'spans_key': spans_key}

    def predict(self, docs: Iterable[Doc]):
        if False:
            while True:
                i = 10
        "Apply the pipeline's model to a batch of docs, without modifying\n        them.\n\n        docs (Iterable[Doc]): The documents to predict.\n        RETURNS: The models prediction for each document.\n\n        DOCS: https://spacy.io/api/spanfinder#predict\n        "
        scores = self.model.predict(docs)
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        if False:
            print('Hello World!')
        'Modify a batch of Doc objects, using pre-computed scores.\n        docs (Iterable[Doc]): The documents to modify.\n        scores: The scores to set, produced by SpanFinder predict method.\n\n        DOCS: https://spacy.io/api/spanfinder#set_annotations\n        '
        offset = 0
        for (i, doc) in enumerate(docs):
            doc.spans[self.cfg['spans_key']] = []
            starts = []
            ends = []
            doc_scores = scores[offset:offset + len(doc)]
            for (token, token_score) in zip(doc, doc_scores):
                if token_score[0] >= self.cfg['threshold']:
                    starts.append(token.i)
                if token_score[1] >= self.cfg['threshold']:
                    ends.append(token.i)
            for start in starts:
                for end in ends:
                    span_length = end + 1 - start
                    if span_length < 1:
                        continue
                    if (self.cfg['min_length'] is None or self.cfg['min_length'] <= span_length) and (self.cfg['max_length'] is None or span_length <= self.cfg['max_length']):
                        doc.spans[self.cfg['spans_key']].append(doc[start:end + 1])
            offset += len(doc)

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        "Learn from a batch of documents and gold-standard information,\n        updating the pipe's model. Delegates to predict and get_loss.\n        examples (Iterable[Example]): A batch of Example objects.\n        drop (float): The dropout rate.\n        sgd (Optional[thinc.api.Optimizer]): The optimizer.\n        losses (Optional[Dict[str, float]]): Optional record of the loss during\n            training. Updated using the component name as the key.\n        RETURNS (Dict[str, float]): The updated losses dictionary.\n\n        DOCS: https://spacy.io/api/spanfinder#update\n        "
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        predicted = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        (scores, backprop_scores) = self.model.begin_update(predicted)
        (loss, d_scores) = self.get_loss(examples, scores)
        backprop_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, examples, scores) -> Tuple[float, Floats2d]:
        if False:
            for i in range(10):
                print('nop')
        "Find the loss and gradient of loss for the batch of documents and\n        their predicted scores.\n        examples (Iterable[Examples]): The batch of examples.\n        scores: Scores representing the model's predictions.\n        RETURNS (Tuple[float, Floats2d]): The loss and the gradient.\n\n        DOCS: https://spacy.io/api/spanfinder#get_loss\n        "
        (truths, masks) = self._get_aligned_truth_scores(examples, self.model.ops)
        d_scores = scores - self.model.ops.asarray2f(truths)
        d_scores *= masks
        loss = float((d_scores ** 2).sum())
        return (loss, d_scores)

    def _get_aligned_truth_scores(self, examples, ops) -> Tuple[Floats2d, Floats2d]:
        if False:
            i = 10
            return i + 15
        'Align scores of the predictions to the references for calculating\n        the loss.\n        '
        truths = []
        masks = []
        for eg in examples:
            if eg.x.text != eg.y.text:
                raise ValueError(Errors.E1054.format(component='span_finder'))
            n_tokens = len(eg.predicted)
            truth = ops.xp.zeros((n_tokens, 2), dtype='float32')
            mask = ops.xp.ones((n_tokens, 2), dtype='float32')
            if self.cfg['spans_key'] in eg.reference.spans:
                for span in eg.reference.spans[self.cfg['spans_key']]:
                    (ref_start_char, ref_end_char) = _char_indices(span)
                    pred_span = eg.predicted.char_span(ref_start_char, ref_end_char, alignment_mode='expand')
                    (pred_start_char, pred_end_char) = _char_indices(pred_span)
                    start_match = pred_start_char == ref_start_char
                    end_match = pred_end_char == ref_end_char
                    if start_match:
                        truth[pred_span[0].i, 0] = 1
                    else:
                        mask[pred_span[0].i, 0] = 0
                    if end_match:
                        truth[pred_span[-1].i, 1] = 1
                    else:
                        mask[pred_span[-1].i, 1] = 0
            truths.append(truth)
            masks.append(mask)
        truths = ops.xp.concatenate(truths, axis=0)
        masks = ops.xp.concatenate(masks, axis=0)
        return (truths, masks)

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the pipe for training, using a representative set\n        of data examples.\n        get_examples (Callable[[], Iterable[Example]]): Function that\n            returns a representative sample of gold-standard Example objects.\n        nlp (Optional[Language]): The current nlp object the component is part\n            of.\n\n        DOCS: https://spacy.io/api/spanfinder#initialize\n        '
        subbatch: List[Example] = []
        for eg in get_examples():
            if len(subbatch) < 10:
                subbatch.append(eg)
        if subbatch:
            docs = [eg.reference for eg in subbatch]
            (Y, _) = self._get_aligned_truth_scores(subbatch, self.model.ops)
            self.model.initialize(X=docs, Y=Y)
        else:
            self.model.initialize()