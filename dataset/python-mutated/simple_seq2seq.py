from typing import Dict
import numpy
from overrides import overrides
import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum

@Model.register('simple_seq2seq')
class SimpleSeq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    This ``SimpleSeq2Seq`` model takes an encoder (:class:`Seq2SeqEncoder`) as an input, and
    implements the functionality of the decoder.  In this implementation, the decoder uses the
    encoder's outputs in two ways. The hidden state of the decoder is intialized with the output
    from the final time-step of the encoder, and when using attention, a weighted average of the
    outputs from the encoder is concatenated to the inputs of the decoder at every timestep.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """

    def __init__(self, vocab: Vocabulary, source_embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, max_decoding_steps: int, target_namespace: str='tokens', target_embedding_dim: int=None, attention_function: SimilarityFunction=None, scheduled_sampling_ratio: float=0.0) -> None:
        if False:
            i = 10
            return i + 15
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = Attention(self._attention_function)
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    @overrides
    def forward(self, source_tokens: Dict[str, torch.LongTensor], target_tokens: Dict[str, torch.LongTensor]=None) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Decoder logic for producing the entire target sequence.\n\n        Parameters\n        ----------\n        source_tokens : Dict[str, torch.LongTensor]\n           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be\n           passed through a ``TextFieldEmbedder`` and then through an encoder.\n        target_tokens : Dict[str, torch.LongTensor], optional (default = None)\n           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the\n           target tokens are also represented as a ``TextField``.\n        '
        embedded_input = self._source_embedder(source_tokens)
        (batch_size, _, _) = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]
        if target_tokens:
            targets = target_tokens['tokens']
            target_sequence_length = targets.size()[1]
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        decoder_context = Variable(encoder_outputs.data.new().resize_(batch_size, self._decoder_output_dim).fill_(0))
        last_predictions = None
        step_logits = []
        step_probabilities = []
        step_predictions = []
        for timestep in range(num_decoding_steps):
            if self.training and all(torch.rand(1) >= self._scheduled_sampling_ratio):
                input_choices = targets[:, timestep]
            elif timestep == 0:
                input_choices = Variable(source_mask.data.new().resize_(batch_size).fill_(self._start_index))
            else:
                input_choices = last_predictions
            decoder_input = self._prepare_decode_step_input(input_choices, decoder_hidden, encoder_outputs, source_mask)
            (decoder_hidden, decoder_context) = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))
            output_projections = self._output_projection_layer(decoder_hidden)
            step_logits.append(output_projections.unsqueeze(1))
            class_probabilities = F.softmax(output_projections)
            (_, predicted_classes) = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {'logits': logits, 'class_probabilities': class_probabilities, 'predictions': all_predictions}
        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict['loss'] = loss
        return output_dict

    def _prepare_decode_step_input(self, input_indices: torch.LongTensor, decoder_hidden_state: torch.LongTensor=None, encoder_outputs: torch.LongTensor=None, encoder_outputs_mask: torch.LongTensor=None) -> torch.LongTensor:
        if False:
            while True:
                i = 10
        "\n        Given the input indices for the current timestep of the decoder, and all the encoder\n        outputs, compute the input at the current timestep.  Note: This method is agnostic to\n        whether the indices are gold indices or the predictions made by the decoder at the last\n        timestep. So, this can be used even if we're doing some kind of scheduled sampling.\n\n        If we're not using attention, the output of this method is just an embedding of the input\n        indices.  If we are, the output will be a concatentation of the embedding and an attended\n        average of the encoder inputs.\n\n        Parameters\n        ----------\n        input_indices : torch.LongTensor\n            Indices of either the gold inputs to the decoder or the predicted labels from the\n            previous timestep.\n        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)\n            Output of from the decoder at the last time step. Needed only if using attention.\n        encoder_outputs : torch.LongTensor, optional (not needed if no attention)\n            Encoder outputs from all time steps. Needed only if using attention.\n        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)\n            Masks on encoder outputs. Needed only if using attention.\n        "
        embedded_input = self._target_embedder(input_indices)
        if self._attention_function:
            encoder_outputs_mask = encoder_outputs_mask.type(torch.FloatTensor)
            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor) -> torch.LongTensor:
        if False:
            i = 10
            return i + 15
        '\n        Takes logits (unnormalized outputs from the decoder) of size (batch_size,\n        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)\n        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross\n        entropy loss while taking the mask into account.\n\n        The length of ``targets`` is expected to be greater than that of ``logits`` because the\n        decoder does not need to compute the output corresponding to the last timestep of\n        ``targets``. This method aligns the inputs appropriately to compute the loss.\n\n        During training, we want the logit corresponding to timestep i to be similar to the target\n        token from timestep i + 1. That is, the targets should be shifted by one timestep for\n        appropriate comparison.  Consider a single example where the target has 3 words, and\n        padding is to 7 tokens.\n           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>\n           and the mask would be                     1   1   1   1   1   0   0\n           and let the logits be                     l1  l2  l3  l4  l5  l6\n        We actually need to compare:\n           the sequence           w1  w2  w3  <E> <P> <P>\n           with masks             1   1   1   1   0   0\n           against                l1  l2  l3  l4  l5  l6\n           (where the input was)  <S> w1  w2  w3  <E> <P>\n        '
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        '\n        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test\n        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives\n        within the ``forward`` method.\n\n        This method trims the output predictions to the first end symbol, replaces indices with\n        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.\n        '
        predicted_indices = output_dict['predictions']
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace='target_tokens') for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        if len(all_predicted_tokens) == 1:
            all_predicted_tokens = all_predicted_tokens[0]
        output_dict['predicted_tokens'] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        if False:
            print('Hello World!')
        source_embedder_params = params.pop('source_embedder')
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop('encoder'))
        max_decoding_steps = params.pop('max_decoding_steps')
        target_namespace = params.pop('target_namespace', 'tokens')
        attention_function_type = params.pop('attention_function', None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop('scheduled_sampling_ratio', 0.0)
        return cls(vocab, source_embedder=source_embedder, encoder=encoder, max_decoding_steps=max_decoding_steps, target_namespace=target_namespace, attention_function=attention_function, scheduled_sampling_ratio=scheduled_sampling_ratio)