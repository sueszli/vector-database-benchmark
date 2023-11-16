from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from ..errors import Errors
from ..language import Language
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
default_model_config = '\n[model]\n@architectures = "spacy.HashEmbedCNN.v2"\npretrained_vectors = null\nwidth = 96\ndepth = 4\nembed_size = 2000\nwindow_size = 1\nmaxout_pieces = 3\nsubword_features = true\n'
DEFAULT_TOK2VEC_MODEL = Config().from_str(default_model_config)['model']

@Language.factory('tok2vec', assigns=['doc.tensor'], default_config={'model': DEFAULT_TOK2VEC_MODEL})
def make_tok2vec(nlp: Language, name: str, model: Model) -> 'Tok2Vec':
    if False:
        while True:
            i = 10
    return Tok2Vec(nlp.vocab, model, name)

class Tok2Vec(TrainablePipe):
    """Apply a "token-to-vector" model and set its outputs in the doc.tensor
    attribute. This is mostly useful to share a single subnetwork between multiple
    components, e.g. to have one embedding and CNN network shared between a
    parser, tagger and NER.

    In order to use the `Tok2Vec` predictions, subsequent components should use
    the `Tok2VecListener` layer as the tok2vec subnetwork of their model. This
    layer will read data from the `doc.tensor` attribute during prediction.
    During training, the `Tok2Vec` component will save its prediction and backprop
    callback for each batch, so that the subsequent components can backpropagate
    to the shared weights. This implementation is used because it allows us to
    avoid relying on object identity within the models to achieve the parameter
    sharing.
    """

    def __init__(self, vocab: Vocab, model: Model, name: str='tok2vec') -> None:
        if False:
            i = 10
            return i + 15
        'Initialize a tok2vec component.\n\n        vocab (Vocab): The shared vocabulary.\n        model (thinc.api.Model[List[Doc], List[Floats2d]]):\n            The Thinc Model powering the pipeline component. It should take\n            a list of Doc objects as input, and output a list of 2d float arrays.\n        name (str): The component instance name.\n\n        DOCS: https://spacy.io/api/tok2vec#init\n        '
        self.vocab = vocab
        self.model = model
        self.name = name
        self.listener_map: Dict[str, List['Tok2VecListener']] = {}
        self.cfg: Dict[str, Any] = {}

    @property
    def listeners(self) -> List['Tok2VecListener']:
        if False:
            return 10
        'RETURNS (List[Tok2VecListener]): The listener models listening to this\n        component. Usually internals.\n        '
        return [m for c in self.listening_components for m in self.listener_map[c]]

    @property
    def listening_components(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'RETURNS (List[str]): The downstream components listening to this\n        component. Usually internals.\n        '
        return list(self.listener_map.keys())

    def add_listener(self, listener: 'Tok2VecListener', component_name: str) -> None:
        if False:
            while True:
                i = 10
        'Add a listener for a downstream component. Usually internals.'
        self.listener_map.setdefault(component_name, [])
        if listener not in self.listener_map[component_name]:
            self.listener_map[component_name].append(listener)

    def remove_listener(self, listener: 'Tok2VecListener', component_name: str) -> bool:
        if False:
            print('Hello World!')
        'Remove a listener for a downstream component. Usually internals.'
        if component_name in self.listener_map:
            if listener in self.listener_map[component_name]:
                self.listener_map[component_name].remove(listener)
                if not self.listener_map[component_name]:
                    del self.listener_map[component_name]
                return True
        return False

    def find_listeners(self, component) -> None:
        if False:
            while True:
                i = 10
        "Walk over a model of a processing component, looking for layers that\n        are Tok2vecListener subclasses that have an upstream_name that matches\n        this component. Listeners can also set their upstream_name attribute to\n        the wildcard string '*' to match any `Tok2Vec`.\n\n        You're unlikely to ever need multiple `Tok2Vec` components, so it's\n        fine to leave your listeners upstream_name on '*'.\n        "
        names = ('*', self.name)
        if isinstance(getattr(component, 'model', None), Model):
            for node in component.model.walk():
                if isinstance(node, Tok2VecListener) and node.upstream_name in names:
                    self.add_listener(node, component.name)

    def predict(self, docs: Iterable[Doc]):
        if False:
            for i in range(10):
                print('nop')
        "Apply the pipeline's model to a batch of docs, without modifying them.\n        Returns a single tensor for a batch of documents.\n\n        docs (Iterable[Doc]): The documents to predict.\n        RETURNS: Vector representations for each token in the documents.\n\n        DOCS: https://spacy.io/api/tok2vec#predict\n        "
        if not any((len(doc) for doc in docs)):
            width = self.model.get_dim('nO')
            return [self.model.ops.alloc((0, width)) for doc in docs]
        tokvecs = self.model.predict(docs)
        return tokvecs

    def set_annotations(self, docs: Sequence[Doc], tokvecses) -> None:
        if False:
            i = 10
            return i + 15
        'Modify a batch of documents, using pre-computed scores.\n\n        docs (Iterable[Doc]): The documents to modify.\n        tokvecses: The tensors to set, produced by Tok2Vec.predict.\n\n        DOCS: https://spacy.io/api/tok2vec#set_annotations\n        '
        for (doc, tokvecs) in zip(docs, tokvecses):
            assert tokvecs.shape[0] == len(doc)
            doc.tensor = tokvecs

    def update(self, examples: Iterable[Example], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None):
        if False:
            i = 10
            return i + 15
        "Learn from a batch of documents and gold-standard information,\n        updating the pipe's model.\n\n        examples (Iterable[Example]): A batch of Example objects.\n        drop (float): The dropout rate.\n        sgd (thinc.api.Optimizer): The optimizer.\n        losses (Dict[str, float]): Optional record of the loss during training.\n            Updated using the component name as the key.\n        RETURNS (Dict[str, float]): The updated losses dictionary.\n\n        DOCS: https://spacy.io/api/tok2vec#update\n        "
        if losses is None:
            losses = {}
        validate_examples(examples, 'Tok2Vec.update')
        docs = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        (tokvecs, bp_tokvecs) = self.model.begin_update(docs)
        d_tokvecs = [self.model.ops.alloc2f(*t2v.shape) for t2v in tokvecs]
        losses.setdefault(self.name, 0.0)

        def accumulate_gradient(one_d_tokvecs):
            if False:
                return 10
            'Accumulate tok2vec loss and gradient. This is passed as a callback\n            to all but the last listener. Only the last one does the backprop.\n            '
            nonlocal d_tokvecs
            for i in range(len(one_d_tokvecs)):
                d_tokvecs[i] += one_d_tokvecs[i]
                losses[self.name] += float((one_d_tokvecs[i] ** 2).sum())
            return [self.model.ops.alloc2f(*t2v.shape) for t2v in tokvecs]

        def backprop(one_d_tokvecs):
            if False:
                for i in range(10):
                    print('nop')
            'Callback to actually do the backprop. Passed to last listener.'
            accumulate_gradient(one_d_tokvecs)
            d_docs = bp_tokvecs(d_tokvecs)
            if sgd is not None:
                self.finish_update(sgd)
            return d_docs
        batch_id = Tok2VecListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, tokvecs, accumulate_gradient)
        if self.listeners:
            self.listeners[-1].receive(batch_id, tokvecs, backprop)
        return losses

    def get_loss(self, examples, scores) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Optional[Language]=None):
        if False:
            return 10
        'Initialize the pipe for training, using a representative set\n        of data examples.\n\n        get_examples (Callable[[], Iterable[Example]]): Function that\n            returns a representative sample of gold-standard Example objects.\n        nlp (Language): The current nlp object the component is part of.\n\n        DOCS: https://spacy.io/api/tok2vec#initialize\n        '
        validate_get_examples(get_examples, 'Tok2Vec.initialize')
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert doc_sample, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def add_label(self, label):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class Tok2VecListener(Model):
    """A layer that gets fed its answers from an upstream connection,
    for instance from a component earlier in the pipeline.

    The Tok2VecListener layer is used as a sublayer within a component such
    as a parser, NER or text categorizer. Usually you'll have multiple listeners
    connecting to a single upstream Tok2Vec component, that's earlier in the
    pipeline. The Tok2VecListener layers act as proxies, passing the predictions
    from the Tok2Vec component into downstream components, and communicating
    gradients back upstream.
    """
    name = 'tok2vec-listener'

    def __init__(self, upstream_name: str, width: int) -> None:
        if False:
            print('Hello World!')
        "\n        upstream_name (str): A string to identify the 'upstream' Tok2Vec component\n            to communicate with. The upstream name should either be the wildcard\n            string '*', or the name of the `Tok2Vec` component. You'll almost\n            never have multiple upstream Tok2Vec components, so the wildcard\n            string will almost always be fine.\n        width (int):\n            The width of the vectors produced by the upstream tok2vec component.\n        "
        Model.__init__(self, name=self.name, forward=forward, dims={'nO': width})
        self.upstream_name = upstream_name
        self._batch_id: Optional[int] = None
        self._outputs = None
        self._backprop = None

    @classmethod
    def get_batch_id(cls, inputs: Iterable[Doc]) -> int:
        if False:
            while True:
                i = 10
        'Calculate a content-sensitive hash of the batch of documents, to check\n        whether the next batch of documents is unexpected.\n        '
        return sum((sum((token.orth for token in doc)) for doc in inputs))

    def receive(self, batch_id: int, outputs, backprop) -> None:
        if False:
            print('Hello World!')
        "Store a batch of training predictions and a backprop callback. The\n        predictions and callback are produced by the upstream Tok2Vec component,\n        and later will be used when the listener's component's model is called.\n        "
        self._batch_id = batch_id
        self._outputs = outputs
        self._backprop = backprop

    def verify_inputs(self, inputs) -> bool:
        if False:
            i = 10
            return i + 15
        'Check that the batch of Doc objects matches the ones we have a\n        prediction for.\n        '
        if self._batch_id is None and self._outputs is None:
            raise ValueError(Errors.E954)
        else:
            batch_id = self.get_batch_id(inputs)
            if batch_id != self._batch_id:
                raise ValueError(Errors.E953.format(id1=batch_id, id2=self._batch_id))
            else:
                return True

def forward(model: Tok2VecListener, inputs, is_train: bool):
    if False:
        i = 10
        return i + 15
    'Supply the outputs from the upstream Tok2Vec component.'
    if is_train:
        if model._batch_id is None:
            outputs = []
            for doc in inputs:
                if doc.tensor.size == 0:
                    raise ValueError(Errors.E203.format(name='tok2vec'))
                else:
                    outputs.append(doc.tensor)
            return (outputs, _empty_backprop)
        else:
            model.verify_inputs(inputs)
            return (model._outputs, model._backprop)
    else:
        outputs = []
        width = model.get_dim('nO')
        for doc in inputs:
            if doc.tensor.size == 0:
                outputs.append(model.ops.alloc2f(len(doc), width))
            else:
                outputs.append(doc.tensor)
        return (outputs, _empty_backprop)

def _empty_backprop(dX):
    if False:
        while True:
            i = 10
    return []