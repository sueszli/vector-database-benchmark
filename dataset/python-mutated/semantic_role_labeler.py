from typing import Dict, List, TextIO, Optional
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

@Model.register('srl')
class SemanticRoleLabeler(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, stacked_encoder: Seq2SeqEncoder, binary_feature_dim: int, embedding_dropout: float=0.0, initializer: InitializerApplicator=InitializerApplicator(), regularizer: Optional[RegularizerApplicator]=None) -> None:
        if False:
            print('Hello World!')
        super(SemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace='labels', ignore_classes=['V'])
        self.stacked_encoder = stacked_encoder
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(), self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        if text_field_embedder.get_output_dim() + binary_feature_dim != stacked_encoder.get_input_dim():
            raise ConfigurationError('The SRL Model uses a binary verb indicator feature, meaning the input dimension of the stacked_encoder must be equal to the output dimension of the text_field_embedder + 1.')
        initializer(self)

    def forward(self, tokens: Dict[str, torch.LongTensor], verb_indicator: torch.LongTensor, tags: torch.LongTensor=None) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        tokens : Dict[str, torch.LongTensor], required\n            The output of ``TextField.as_array()``, which should typically be passed directly to a\n            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``\n            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":\n            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used\n            for the ``TokenIndexers`` when you created the ``TextField`` representing your\n            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,\n            which knows how to combine different word representations into a single vector per\n            token in your input.\n        verb_indicator: torch.LongTensor, required.\n            An integer ``SequenceFeatureField`` representation of the position of the verb\n            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be\n            all zeros, in the case that the sentence has no verbal predicate.\n        tags : torch.LongTensor, optional (default = None)\n            A torch tensor representing the sequence of integer gold class labels\n            of shape ``(batch_size, num_tokens)``\n\n        Returns\n        -------\n        An output dictionary consisting of:\n        logits : torch.FloatTensor\n            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing\n            unnormalised log probabilities of the tag classes.\n        class_probabilities : torch.FloatTensor\n            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing\n            a distribution of the tag classes per word.\n        loss : torch.FloatTensor, optional\n            A scalar loss to be optimised.\n\n        '
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        (batch_size, sequence_length, embedding_dim_with_binary_feature) = embedded_text_with_verb_indicator.size()
        if self.stacked_encoder.get_input_dim() != embedding_dim_with_binary_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes the embedding dimension one larger than the value specified. Therefore, the 'input_dim' of the stacked_encoder must be equal to total_embedding_dim + 1.")
        encoded_text = self.stacked_encoder(embedded_text_with_verb_indicator, mask)
        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, sequence_length, self.num_classes])
        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            self.span_metric(class_probabilities, tags, mask)
            output_dict['loss'] = loss
        output_dict['mask'] = mask
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The\n        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a\n        ``"tags"`` key to the dictionary with the result.\n        '
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for (predictions, length) in zip(predictions_list, sequence_lengths):
            (max_likelihood_sequence, _) = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace='labels') for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool=False):
        if False:
            while True:
                i = 10
        metric_dict = self.span_metric.get_metric(reset=reset)
        if self.training:
            return {x: y for (x, y) in metric_dict.items() if 'overall' in x}
        return metric_dict

    def get_viterbi_pairwise_potentials(self):
        if False:
            return 10
        '\n        Generate a matrix of pairwise transition potentials for the BIO labels.\n        The only constraint implemented here is that I-XXX labels must be preceded\n        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this\n        constraint, pairs of labels which do not satisfy this constraint have a\n        pairwise potential of -inf.\n\n        Returns\n        -------\n        transition_matrix : torch.Tensor\n            A (num_labels, num_labels) matrix of pairwise potentials.\n        '
        all_labels = self.vocab.get_index_to_token_vocabulary('labels')
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])
        for (i, previous_label) in all_labels.items():
            for (j, label) in all_labels.items():
                if i != j and label[0] == 'I' and (not previous_label == 'B' + label[1:]):
                    transition_matrix[i, j] = float('-inf')
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SemanticRoleLabeler':
        if False:
            while True:
                i = 10
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop('stacked_encoder'))
        binary_feature_dim = params.pop('binary_feature_dim')
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab=vocab, text_field_embedder=text_field_embedder, stacked_encoder=stacked_encoder, binary_feature_dim=binary_feature_dim, initializer=initializer, regularizer=regularizer)

def write_to_conll_eval_file(prediction_file: TextIO, gold_file: TextIO, verb_index: Optional[int], sentence: List[str], prediction: List[str], gold_labels: List[str]):
    if False:
        while True:
            i = 10
    '\n    Prints predicate argument predictions and gold labels for a single verbal\n    predicate in a sentence to two provided file references.\n\n    Parameters\n    ----------\n    prediction_file : TextIO, required.\n        A file reference to print predictions to.\n    gold_file : TextIO, required.\n        A file reference to print gold labels to.\n    verb_index : Optional[int], required.\n        The index of the verbal predicate in the sentence which\n        the gold labels are the arguments for, or None if the sentence\n        contains no verbal predicate.\n    sentence : List[str], required.\n        The word tokens.\n    prediction : List[str], required.\n        The predicted BIO labels.\n    gold_labels : List[str], required.\n        The gold BIO labels.\n    '
    verb_only_sentence = ['-'] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]
    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)
    for (word, predicted, gold) in zip(verb_only_sentence, conll_format_predictions, conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + '\n')
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + '\n')
    prediction_file.write('\n')
    gold_file.write('\n')

def convert_bio_tags_to_conll_format(labels: List[str]):
    if False:
        while True:
            i = 10
    '\n    Converts BIO formatted SRL tags to the format required for evaluation with the\n    official CONLL 2005 perl script. Spans are represented by bracketed labels,\n    with the labels of words inside spans being the same as those outside spans.\n    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )\n    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for\n    length 1 spans, (e.g "(ARG-0*)").\n\n    A full example of the conversion performed:\n\n    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]\n    [ "(ARG-1*", "*", "*", "*", "*)", "*"]\n\n    Parameters\n    ----------\n    labels : List[str], required.\n        A list of BIO tags to convert to the CONLL span based format.\n\n    Returns\n    -------\n    A list of labels in the CONLL span based format.\n    '
    sentence_length = len(labels)
    conll_labels = []
    for (i, label) in enumerate(labels):
        if label == 'O':
            conll_labels.append('*')
            continue
        new_label = '*'
        if label[0] == 'B' or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = '(' + label[2:] + new_label
        if i == sentence_length - 1 or labels[i + 1][0] == 'B' or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ')'
        conll_labels.append(new_label)
    return conll_labels