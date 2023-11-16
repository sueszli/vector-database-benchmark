from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy
from thinc.api import Config, Model, Optimizer, get_array_module, set_dropout_rate
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
single_label_default_config = '\n[model]\n@architectures = "spacy.TextCatEnsemble.v2"\n\n[model.tok2vec]\n@architectures = "spacy.Tok2Vec.v2"\n\n[model.tok2vec.embed]\n@architectures = "spacy.MultiHashEmbed.v2"\nwidth = 64\nrows = [2000, 2000, 500, 1000, 500]\nattrs = ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"]\ninclude_static_vectors = false\n\n[model.tok2vec.encode]\n@architectures = "spacy.MaxoutWindowEncoder.v2"\nwidth = ${model.tok2vec.embed.width}\nwindow_size = 1\nmaxout_pieces = 3\ndepth = 2\n\n[model.linear_model]\n@architectures = "spacy.TextCatBOW.v2"\nexclusive_classes = true\nngram_size = 1\nno_output_layer = false\n'
DEFAULT_SINGLE_TEXTCAT_MODEL = Config().from_str(single_label_default_config)['model']
single_label_bow_config = '\n[model]\n@architectures = "spacy.TextCatBOW.v2"\nexclusive_classes = true\nngram_size = 1\nno_output_layer = false\n'
single_label_cnn_config = '\n[model]\n@architectures = "spacy.TextCatCNN.v2"\nexclusive_classes = true\n\n[model.tok2vec]\n@architectures = "spacy.HashEmbedCNN.v2"\npretrained_vectors = null\nwidth = 96\ndepth = 4\nembed_size = 2000\nwindow_size = 1\nmaxout_pieces = 3\nsubword_features = true\n'

@Language.factory('textcat', assigns=['doc.cats'], default_config={'threshold': 0.0, 'model': DEFAULT_SINGLE_TEXTCAT_MODEL, 'scorer': {'@scorers': 'spacy.textcat_scorer.v2'}}, default_score_weights={'cats_score': 1.0, 'cats_score_desc': None, 'cats_micro_p': None, 'cats_micro_r': None, 'cats_micro_f': None, 'cats_macro_p': None, 'cats_macro_r': None, 'cats_macro_f': None, 'cats_macro_auc': None, 'cats_f_per_type': None})
def make_textcat(nlp: Language, name: str, model: Model[List[Doc], List[Floats2d]], threshold: float, scorer: Optional[Callable]) -> 'TextCategorizer':
    if False:
        while True:
            i = 10
    'Create a TextCategorizer component. The text categorizer predicts categories\n    over a whole document. It can learn one or more labels, and the labels are considered\n    to be mutually exclusive (i.e. one true label per doc).\n\n    model (Model[List[Doc], List[Floats2d]]): A model instance that predicts\n        scores for each category.\n    threshold (float): Cutoff to consider a prediction "positive".\n    scorer (Optional[Callable]): The scoring method.\n    '
    return TextCategorizer(nlp.vocab, model, name, threshold=threshold, scorer=scorer)

def textcat_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    return Scorer.score_cats(examples, 'cats', multi_label=False, **kwargs)

@registry.scorers('spacy.textcat_scorer.v2')
def make_textcat_scorer():
    if False:
        for i in range(10):
            print('nop')
    return textcat_score

class TextCategorizer(TrainablePipe):
    """Pipeline component for single-label text classification.

    DOCS: https://spacy.io/api/textcategorizer
    """

    def __init__(self, vocab: Vocab, model: Model, name: str='textcat', *, threshold: float, scorer: Optional[Callable]=textcat_score) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a text categorizer for single-label classification.\n\n        vocab (Vocab): The shared vocabulary.\n        model (thinc.api.Model): The Thinc Model powering the pipeline component.\n        name (str): The component instance name, used to add entries to the\n            losses during training.\n        threshold (float): Unused, not needed for single-label (exclusive\n            classes) classification.\n        scorer (Optional[Callable]): The scoring method. Defaults to\n                Scorer.score_cats for the attribute "cats".\n\n        DOCS: https://spacy.io/api/textcategorizer#init\n        '
        self.vocab = vocab
        self.model = model
        self.name = name
        self._rehearsal_model = None
        cfg: Dict[str, Any] = {'labels': [], 'threshold': threshold, 'positive_label': None}
        self.cfg = dict(cfg)
        self.scorer = scorer

    @property
    def support_missing_values(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    @property
    def labels(self) -> Tuple[str]:
        if False:
            while True:
                i = 10
        'RETURNS (Tuple[str]): The labels currently added to the component.\n\n        DOCS: https://spacy.io/api/textcategorizer#labels\n        '
        return tuple(self.cfg['labels'])

    @property
    def label_data(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        "RETURNS (List[str]): Information about the component's labels.\n\n        DOCS: https://spacy.io/api/textcategorizer#label_data\n        "
        return self.labels

    def predict(self, docs: Iterable[Doc]):
        if False:
            while True:
                i = 10
        "Apply the pipeline's model to a batch of docs, without modifying them.\n\n        docs (Iterable[Doc]): The documents to predict.\n        RETURNS: The models prediction for each document.\n\n        DOCS: https://spacy.io/api/textcategorizer#predict\n        "
        if not any((len(doc) for doc in docs)):
            tensors = [doc.tensor for doc in docs]
            xp = self.model.ops.xp
            scores = xp.zeros((len(list(docs)), len(self.labels)))
            return scores
        scores = self.model.predict(docs)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores) -> None:
        if False:
            return 10
        'Modify a batch of Doc objects, using pre-computed scores.\n\n        docs (Iterable[Doc]): The documents to modify.\n        scores: The scores to set, produced by TextCategorizer.predict.\n\n        DOCS: https://spacy.io/api/textcategorizer#set_annotations\n        '
        for (i, doc) in enumerate(docs):
            for (j, label) in enumerate(self.labels):
                doc.cats[label] = float(scores[i, j])

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        if False:
            while True:
                i = 10
        "Learn from a batch of documents and gold-standard information,\n        updating the pipe's model. Delegates to predict and get_loss.\n\n        examples (Iterable[Example]): A batch of Example objects.\n        drop (float): The dropout rate.\n        sgd (thinc.api.Optimizer): The optimizer.\n        losses (Dict[str, float]): Optional record of the loss during training.\n            Updated using the component name as the key.\n        RETURNS (Dict[str, float]): The updated losses dictionary.\n\n        DOCS: https://spacy.io/api/textcategorizer#update\n        "
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, 'TextCategorizer.update')
        self._validate_categories(examples)
        if not any((len(eg.predicted) if eg.predicted else 0 for eg in examples)):
            return losses
        set_dropout_rate(self.model, drop)
        (scores, bp_scores) = self.model.begin_update([eg.predicted for eg in examples])
        (loss, d_scores) = self.get_loss(examples, scores)
        bp_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def rehearse(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        if False:
            return 10
        'Perform a "rehearsal" update from a batch of data. Rehearsal updates\n        teach the current model to make predictions similar to an initial model,\n        to try to address the "catastrophic forgetting" problem. This feature is\n        experimental.\n\n        examples (Iterable[Example]): A batch of Example objects.\n        drop (float): The dropout rate.\n        sgd (thinc.api.Optimizer): The optimizer.\n        losses (Dict[str, float]): Optional record of the loss during training.\n            Updated using the component name as the key.\n        RETURNS (Dict[str, float]): The updated losses dictionary.\n\n        DOCS: https://spacy.io/api/textcategorizer#rehearse\n        '
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        if self._rehearsal_model is None:
            return losses
        validate_examples(examples, 'TextCategorizer.rehearse')
        self._validate_categories(examples)
        docs = [eg.predicted for eg in examples]
        if not any((len(doc) for doc in docs)):
            return losses
        set_dropout_rate(self.model, drop)
        (scores, bp_scores) = self.model.begin_update(docs)
        (target, _) = self._rehearsal_model.begin_update(docs)
        gradient = scores - target
        bp_scores(gradient)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += (gradient ** 2).sum()
        return losses

    def _examples_to_truth(self, examples: Iterable[Example]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if False:
            while True:
                i = 10
        nr_examples = len(list(examples))
        truths = numpy.zeros((nr_examples, len(self.labels)), dtype='f')
        not_missing = numpy.ones((nr_examples, len(self.labels)), dtype='f')
        for (i, eg) in enumerate(examples):
            for (j, label) in enumerate(self.labels):
                if label in eg.reference.cats:
                    truths[i, j] = eg.reference.cats[label]
                elif self.support_missing_values:
                    not_missing[i, j] = 0.0
        truths = self.model.ops.asarray(truths)
        return (truths, not_missing)

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        if False:
            for i in range(10):
                print('nop')
        "Find the loss and gradient of loss for the batch of documents and\n        their predicted scores.\n\n        examples (Iterable[Examples]): The batch of examples.\n        scores: Scores representing the model's predictions.\n        RETURNS (Tuple[float, float]): The loss and the gradient.\n\n        DOCS: https://spacy.io/api/textcategorizer#get_loss\n        "
        validate_examples(examples, 'TextCategorizer.get_loss')
        self._validate_categories(examples)
        (truths, not_missing) = self._examples_to_truth(examples)
        not_missing = self.model.ops.asarray(not_missing)
        d_scores = scores - truths
        d_scores *= not_missing
        mean_square_error = (d_scores ** 2).mean()
        return (float(mean_square_error), d_scores)

    def add_label(self, label: str) -> int:
        if False:
            return 10
        'Add a new label to the pipe.\n\n        label (str): The label to add.\n        RETURNS (int): 0 if label is already present, otherwise 1.\n\n        DOCS: https://spacy.io/api/textcategorizer#add_label\n        '
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._allow_extra_label()
        self.cfg['labels'].append(label)
        if self.model and 'resize_output' in self.model.attrs:
            self.model = self.model.attrs['resize_output'](self.model, len(self.labels))
        self.vocab.strings.add(label)
        return 1

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None, labels: Optional[Iterable[str]]=None, positive_label: Optional[str]=None) -> None:
        if False:
            return 10
        'Initialize the pipe for training, using a representative set\n        of data examples.\n\n        get_examples (Callable[[], Iterable[Example]]): Function that\n            returns a representative sample of gold-standard Example objects.\n        nlp (Language): The current nlp object the component is part of.\n        labels (Optional[Iterable[str]]): The labels to add to the component, typically generated by the\n            `init labels` command. If no labels are provided, the get_examples\n            callback is used to extract the labels from the data.\n        positive_label (Optional[str]): The positive label for a binary task with exclusive classes,\n            `None` otherwise and by default.\n\n        DOCS: https://spacy.io/api/textcategorizer#initialize\n        '
        validate_get_examples(get_examples, 'TextCategorizer.initialize')
        self._validate_categories(get_examples())
        if labels is None:
            for example in get_examples():
                for cat in example.y.cats:
                    self.add_label(cat)
        else:
            for label in labels:
                self.add_label(label)
        if len(self.labels) < 2:
            raise ValueError(Errors.E867)
        if positive_label is not None:
            if positive_label not in self.labels:
                err = Errors.E920.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
            if len(self.labels) != 2:
                err = Errors.E919.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
        self.cfg['positive_label'] = positive_label
        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        (label_sample, _) = self._examples_to_truth(subbatch)
        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _validate_categories(self, examples: Iterable[Example]):
        if False:
            print('Hello World!')
        'Check whether the provided examples all have single-label cats annotations.'
        for ex in examples:
            vals = list(ex.reference.cats.values())
            if vals.count(1.0) > 1:
                raise ValueError(Errors.E895.format(value=ex.reference.cats))
            for val in vals:
                if not (val == 1.0 or val == 0.0):
                    raise ValueError(Errors.E851.format(val=val))