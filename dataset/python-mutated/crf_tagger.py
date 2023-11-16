from typing import Dict, Optional
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure

@Model.register('crf_tagger')
class CrfTagger(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, label_namespace: str='labels', initializer: InitializerApplicator=InitializerApplicator(), regularizer: Optional[RegularizerApplicator]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(), self.num_tags))
        self.crf = ConditionalRandomField(self.num_tags)
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)
        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError('The output dimension of the text_field_embedder must match the input dimension of the phrase_encoder. Found {} and {}, respectively.'.format(text_field_embedder.get_output_dim(), encoder.get_input_dim()))
        initializer(self)

    @overrides
    def forward(self, tokens: Dict[str, torch.LongTensor], tags: torch.LongTensor=None) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        tokens : ``Dict[str, torch.LongTensor]``, required\n            The output of ``TextField.as_array()``, which should typically be passed directly to a\n            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``\n            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":\n            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used\n            for the ``TokenIndexers`` when you created the ``TextField`` representing your\n            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,\n            which knows how to combine different word representations into a single vector per\n            token in your input.\n        tags : ``torch.LongTensor``, optional (default = ``None``)\n            A torch tensor representing the sequence of integer gold class labels of shape\n            ``(batch_size, num_tokens)``.\n\n        Returns\n        -------\n        An output dictionary consisting of:\n\n        logits : ``torch.FloatTensor``\n            The logits that are the output of the ``tag_projection_layer``\n        mask : ``torch.LongTensor``\n            The text field mask for the input tokens\n        tags : ``List[List[str]]``\n            The predicted tags using the Viterbi algorithm.\n        loss : ``torch.FloatTensor``, optional\n            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.\n        '
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)
        logits = self.tag_projection_layer(encoded_text)
        predicted_tags = self.crf.viterbi_tags(logits, mask)
        output = {'logits': logits, 'mask': mask, 'tags': predicted_tags}
        if tags is not None:
            log_likelihood = self.crf.forward(logits, tags, mask)
            output['loss'] = -log_likelihood
            class_probabilities = logits * 0.0
            for (i, instance_tags) in enumerate(predicted_tags):
                for (j, tag_id) in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self.span_metric(class_probabilities, tags, mask)
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Converts the tag ids to the actual tags.\n        ``output_dict["tags"]`` is a list of lists of tag_ids,\n        so we use an ugly nested list comprehension.\n        '
        output_dict['tags'] = [[self.vocab.get_token_from_index(tag, namespace='labels') for tag in instance_tags] for instance_tags in output_dict['tags']]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {x: y for (x, y) in metric_dict.items() if 'overall' in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CrfTagger':
        if False:
            for i in range(10):
                print('nop')
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop('encoder'))
        label_namespace = params.pop('label_namespace', 'labels')
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab, text_field_embedder=text_field_embedder, encoder=encoder, label_namespace=label_namespace, initializer=initializer, regularizer=regularizer)