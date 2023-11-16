import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from overrides import overrides
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores
logger = logging.getLogger(__name__)

@Model.register('coref')
class CoreferenceResolver(Model):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, context_layer: Seq2SeqEncoder, mention_feedforward: FeedForward, antecedent_feedforward: FeedForward, feature_size: int, max_span_width: int, spans_per_word: float, max_antecedents: int, lexical_dropout: float=0.2, initializer: InitializerApplicator=InitializerApplicator(), regularizer: Optional[RegularizerApplicator]=None) -> None:
        if False:
            while True:
                i = 10
        super(CoreferenceResolver, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._mention_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1))
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))
        self._head_scorer = TimeDistributed(torch.nn.Linear(context_layer.get_output_dim(), 1))
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)
        self._span_width_embedding = Embedding(max_span_width, feature_size)
        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents
        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor], span_starts: torch.IntTensor, span_ends: torch.IntTensor, span_labels: torch.IntTensor=None, metadata: List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        text : ``Dict[str, torch.LongTensor]``, required.\n            The output of a ``TextField`` representing the text of\n            the document.\n        span_starts : ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans, 1), representing the start indices of\n            candidate spans for mentions. Comes from a ``ListField[IndexField]`` of indices into\n            the text of the document.\n        span_ends : ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans, 1), representing the end indices of\n            candidate spans for mentions. Comes from a ``ListField[IndexField]`` of indices into\n            the text of the document.\n        span_labels : ``torch.IntTensor``, optional (default = None)\n            A tensor of shape (batch_size, num_spans), representing the cluster ids\n            of each span, or -1 for those which do not appear in any clusters.\n\n        Returns\n        -------\n        An output dictionary consisting of:\n        top_spans : ``torch.IntTensor``\n            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing\n            the start and end word indices of the top spans that survived the pruning stage.\n        antecedent_indices : ``torch.IntTensor``\n            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span\n            the index (with respect to top_spans) of the possible antecedents the model considered.\n        predicted_antecedents : ``torch.IntTensor``\n            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the\n            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there\n            was no predicted link.\n        loss : ``torch.FloatTensor``, optional\n            A scalar loss to be optimised.\n        '
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        document_length = text_embeddings.size(1)
        num_spans = span_starts.size(1)
        text_mask = util.get_text_field_mask(text).float()
        span_mask = (span_starts >= 0).float()
        span_starts = F.relu(span_starts.float()).long()
        span_ends = F.relu(span_ends.float()).long()
        span_embeddings = self._compute_span_representations(text_embeddings, text_mask, span_starts, span_ends)
        mention_scores = self._mention_scorer(self._mention_feedforward(span_embeddings))
        mention_scores += span_mask.log()
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))
        top_span_indices = self._prune_and_sort_spans(mention_scores, num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_span_embeddings = util.batched_index_select(span_embeddings, top_span_indices, flat_top_span_indices)
        top_span_mask = util.batched_index_select(span_mask, top_span_indices, flat_top_span_indices)
        top_span_mention_scores = util.batched_index_select(mention_scores, top_span_indices, flat_top_span_indices)
        top_span_starts = util.batched_index_select(span_starts, top_span_indices, flat_top_span_indices)
        top_span_ends = util.batched_index_select(span_ends, top_span_indices, flat_top_span_indices)
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)
        (valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask) = self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, text_mask.is_cuda)
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings, valid_antecedent_indices)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores, valid_antecedent_indices).squeeze(-1)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings, candidate_antecedent_embeddings, valid_antecedent_offsets)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings, top_span_mention_scores, candidate_antecedent_mention_scores, valid_antecedent_log_mask)
        top_spans = torch.cat([top_span_starts, top_span_ends], -1)
        (_, predicted_antecedents) = coreference_scores.max(2)
        predicted_antecedents -= 1
        output_dict = {'top_spans': top_spans, 'antecedent_indices': valid_antecedent_indices, 'predicted_antecedents': predicted_antecedents}
        if span_labels is not None:
            pruned_gold_labels = util.batched_index_select(span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices)
            antecedent_labels = util.flattened_index_select(pruned_gold_labels, valid_antecedent_indices).squeeze(-1)
            antecedent_labels += valid_antecedent_log_mask.long()
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels, antecedent_labels)
            coreference_log_probs = util.last_dim_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()
            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)
            output_dict['loss'] = negative_marginal_log_likelihood
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        if False:
            print('Hello World!')
        '\n        Converts the list of spans and predicted antecedent indices into clusters\n        of spans for each element in the batch.\n\n        Parameters\n        ----------\n        output_dict : ``Dict[str, torch.Tensor]``, required.\n            The result of calling :func:`forward` on an instance or batch of instances.\n\n        Returns\n        -------\n        The same output dictionary, but with an additional ``clusters`` key:\n\n        clusters : ``List[List[List[Tuple[int, int]]]]``\n            A nested list, representing, for each instance in the batch, the list of clusters,\n            which are in turn comprised of a list of (start, end) inclusive spans into the\n            original document.\n        '
        batch_top_spans = output_dict['top_spans'].data.cpu()
        batch_predicted_antecedents = output_dict['predicted_antecedents'].data.cpu()
        antecedent_indices = output_dict['antecedent_indices'].data.cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []
        for (top_spans, predicted_antecedents) in zip(batch_top_spans, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []
            for (i, (span, predicted_antecedent)) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    continue
                predicted_index = antecedent_indices[i, predicted_antecedent]
                antecedent_span = (top_spans[predicted_index, 0], top_spans[predicted_index, 1])
                if antecedent_span in spans_to_cluster_ids.keys():
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    predicted_cluster_id = len(clusters)
                    clusters.append([antecedent_span])
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id
                (span_start, span_end) = span
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[span_start, span_end] = predicted_cluster_id
            batch_clusters.append(clusters)
        output_dict['clusters'] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        if False:
            return 10
        mention_recall = self._mention_recall.get_metric(reset)
        (coref_precision, coref_recall, coref_f1) = self._conll_coref_scores.get_metric(reset)
        return {'coref_precision': coref_precision, 'coref_recall': coref_recall, 'coref_f1': coref_f1, 'mention_recall': mention_recall}

    def _create_attended_span_representations(self, head_scores: torch.FloatTensor, text_embeddings: torch.FloatTensor, span_ends: torch.IntTensor, span_widths: torch.IntTensor) -> torch.FloatTensor:
        if False:
            return 10
        '\n        Given a tensor of unnormalized attention scores for each word in the document, compute\n        distributions over every span with respect to these scores by normalising the headedness\n        scores for words inside the span.\n\n        Given these headedness distributions over every span, weight the corresponding vector\n        representations of the words in the span by this distribution, returning a weighted\n        representation of each span.\n\n        Parameters\n        ----------\n        head_scores : ``torch.FloatTensor``, required.\n            Unnormalized headedness scores for every word. This score is shared for every\n            candidate. The only way in which the headedness scores differ over different\n            spans is in the set of words over which they are normalized.\n        text_embeddings: ``torch.FloatTensor``, required.\n            The embeddings with shape  (batch_size, document_length, embedding_size)\n            over which we are computing a weighted sum.\n        span_ends: ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans, 1), representing the end indices\n            of each span.\n        span_widths : ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans, 1) representing the width of each\n            span candidates.\n        Returns\n        -------\n        attended_text_embeddings : ``torch.FloatTensor``\n            A tensor of shape (batch_size, num_spans, embedding_dim) - the result of\n            applying attention over all words within each candidate span.\n        '
        max_span_range_indices = util.get_range_vector(self._max_span_width, text_embeddings.is_cuda).view(1, 1, -1)
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = F.relu(raw_span_indices.float()).long()
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, text_embeddings.size(1))
        span_text_embeddings = util.batched_index_select(text_embeddings, span_indices, flat_span_indices)
        span_head_scores = util.batched_index_select(head_scores, span_indices, flat_span_indices).squeeze(-1)
        span_head_weights = util.last_dim_softmax(span_head_scores, span_mask)
        attended_text_embeddings = util.weighted_sum(span_text_embeddings, span_head_weights)
        return attended_text_embeddings

    def _compute_span_representations(self, text_embeddings: torch.FloatTensor, text_mask: torch.FloatTensor, span_starts: torch.IntTensor, span_ends: torch.IntTensor) -> torch.FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        "\n        Computes an embedded representation of every candidate span. This is a concatenation\n        of the contextualized endpoints of the span, an embedded representation of the width of\n        the span and a representation of the span's predicted head.\n\n        Parameters\n        ----------\n        text_embeddings : ``torch.FloatTensor``, required.\n            The embedded document of shape (batch_size, document_length, embedding_dim)\n            over which we are computing a weighted sum.\n        text_mask : ``torch.FloatTensor``, required.\n            A mask of shape (batch_size, document_length) representing non-padding entries of\n            ``text_embeddings``.\n        span_starts : ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans) representing the start of each span candidate.\n        span_ends : ``torch.IntTensor``, required.\n            A tensor of shape (batch_size, num_spans) representing the end of each span candidate.\n        Returns\n        -------\n        span_embeddings : ``torch.FloatTensor``\n            An embedded representation of every candidate span with shape:\n            (batch_size, num_spans, context_layer.get_output_dim() * 2 + embedding_size + feature_size)\n        "
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        start_embeddings = util.batched_index_select(contextualized_embeddings, span_starts.squeeze(-1))
        end_embeddings = util.batched_index_select(contextualized_embeddings, span_ends.squeeze(-1))
        span_widths = span_ends - span_starts
        span_width_embeddings = self._span_width_embedding(span_widths.squeeze(-1))
        head_scores = self._head_scorer(contextualized_embeddings)
        attended_text_embeddings = self._create_attended_span_representations(head_scores, text_embeddings, span_ends, span_widths)
        span_embeddings = torch.cat([start_embeddings, end_embeddings, span_width_embeddings, attended_text_embeddings], -1)
        return span_embeddings

    @staticmethod
    def _prune_and_sort_spans(mention_scores: torch.FloatTensor, num_spans_to_keep: int) -> torch.IntTensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        The indices of the top-k scoring spans according to span_scores. We return the\n        indices in their original order, not ordered by score, so that we can rely on\n        the ordering to consider the previous k spans as antecedents for each span later.\n\n        Parameters\n        ----------\n        mention_scores : ``torch.FloatTensor``, required.\n            The mention score for every candidate, with shape (batch_size, num_spans, 1).\n        num_spans_to_keep : ``int``, required.\n            The number of spans to keep when pruning.\n        Returns\n        -------\n        top_span_indices : ``torch.IntTensor``, required.\n            The indices of the top-k scoring spans. Has shape (batch_size, num_spans_to_keep).\n        '
        (_, top_span_indices) = mention_scores.topk(num_spans_to_keep, 1)
        (top_span_indices, _) = torch.sort(top_span_indices, 1)
        top_span_indices = top_span_indices.squeeze(-1)
        return top_span_indices

    @staticmethod
    def _generate_valid_antecedents(num_spans_to_keep: int, max_antecedents: int, is_cuda: bool) -> Tuple[torch.IntTensor, torch.IntTensor, torch.FloatTensor]:
        if False:
            while True:
                i = 10
        '\n        This method generates possible antecedents per span which survived the pruning\n        stage. This procedure is `generic across the batch`. The reason this is the case is\n        that each span in a batch can be coreferent with any previous span, but here we\n        are computing the possible `indices` of these spans. So, regardless of the batch,\n        the 1st span _cannot_ have any antecedents, because there are none to select from.\n        Similarly, each element can only predict previous spans, so this returns a matrix\n        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to\n        (i - 1) - j if j <= i, or zero otherwise.\n\n        Parameters\n        ----------\n        num_spans_to_keep : ``int``, required.\n            The number of spans that were kept while pruning.\n        max_antecedents : ``int``, required.\n            The maximum number of antecedent spans to consider for every span.\n        is_cuda : ``bool``, required.\n            Whether the computation is being done on the GPU or not.\n\n        Returns\n        -------\n        valid_antecedent_indices : ``torch.IntTensor``\n            The indices of every antecedent to consider with respect to the top k spans.\n            Has shape ``(num_spans_to_keep, max_antecedents)``.\n        valid_antecedent_offsets : ``torch.IntTensor``\n            The distance between the span and each of its antecedents in terms of the number\n            of considered spans (i.e not the word distance between the spans).\n            Has shape ``(1, max_antecedents)``.\n        valid_antecedent_log_mask : ``torch.FloatTensor``\n            The logged mask representing whether each antecedent span is valid. Required since\n            different spans have different numbers of valid antecedents. For example, the first\n            span in the document should have no valid antecedents.\n            Has shape ``(1, num_spans_to_keep, max_antecedents)``.\n        '
        target_indices = util.get_range_vector(num_spans_to_keep, is_cuda).unsqueeze(1)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, is_cuda) + 1).unsqueeze(0)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets
        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return (valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask)

    def _compute_span_pair_embeddings(self, top_span_embeddings: torch.FloatTensor, antecedent_embeddings: torch.FloatTensor, antecedent_offsets: torch.FloatTensor):
        if False:
            while True:
                i = 10
        '\n        Computes an embedding representation of pairs of spans for the pairwise scoring function\n        to consider. This includes both the original span representations, the element-wise\n        similarity of the span representations, and an embedding representation of the distance\n        between the two spans.\n\n        Parameters\n        ----------\n        top_span_embeddings : ``torch.FloatTensor``, required.\n            Embedding representations of the top spans. Has shape\n            (batch_size, num_spans_to_keep, embedding_size).\n        antecedent_embeddings : ``torch.FloatTensor``, required.\n            Embedding representations of the antecedent spans we are considering\n            for each top span. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).\n        antecedent_offsets : ``torch.IntTensor``, required.\n            The offsets between each top span and its antecedent spans in terms\n            of spans we are considering. Has shape (1, max_antecedents).\n\n        Returns\n        -------\n        span_pair_embeddings : ``torch.FloatTensor``\n            Embedding representation of the pair of spans to consider. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)\n        '
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)
        antecedent_distance_embeddings = self._distance_embedding(util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets))
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)
        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0), antecedent_embeddings.size(1), antecedent_embeddings.size(2), antecedent_distance_embeddings.size(-1))
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)
        span_pair_embeddings = torch.cat([target_embeddings, antecedent_embeddings, antecedent_embeddings * target_embeddings, antecedent_distance_embeddings], -1)
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_span_labels: torch.IntTensor, antecedent_labels: torch.IntTensor):
        if False:
            i = 10
            return i + 15
        '\n        Generates a binary indicator for every pair of spans. This label is one if and\n        only if the pair of spans belong to the same cluster. The labels are augmented\n        with a dummy antecedent at the zeroth position, which represents the prediction\n        that a span does not have any antecedent.\n\n        Parameters\n        ----------\n        top_span_labels : ``torch.IntTensor``, required.\n            The cluster id label for every span. The id is arbitrary,\n            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).\n        antecedent_labels : ``torch.IntTensor``, required.\n            The cluster id label for every antecedent span. The id is arbitrary,\n            as we just care about the clustering. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents).\n\n        Returns\n        -------\n        pairwise_labels_with_dummy_label : ``torch.FloatTensor``\n            A binary tensor representing whether a given pair of spans belong to\n            the same cluster in the gold clustering.\n            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).\n\n        '
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(self, pairwise_embeddings: torch.FloatTensor, top_span_mention_scores: torch.FloatTensor, antecedent_mention_scores: torch.FloatTensor, antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        '\n        Computes scores for every pair of spans. Additionally, a dummy label is included,\n        representing the decision that the span is not coreferent with anything. For the dummy\n        label, the score is always zero. For the true antecedent spans, the score consists of\n        the pairwise antecedent score and the unary mention scores for the span and its\n        antecedent. The factoring allows the model to blame many of the absent links on bad\n        spans, enabling the pruning strategy used in the forward pass.\n\n        Parameters\n        ----------\n        pairwise_embeddings: ``torch.FloatTensor``, required.\n            Embedding representations of pairs of spans. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)\n        top_span_mention_scores: ``torch.FloatTensor``, required.\n            Mention scores for every span. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents).\n        antecedent_mention_scores: ``torch.FloatTensor``, required.\n            Mention scores for every antecedent. Has shape\n            (batch_size, num_spans_to_keep, max_antecedents).\n        antecedent_log_mask: ``torch.FloatTensor``, required.\n            The log of the mask for valid antecedents.\n\n        Returns\n        -------\n        coreference_scores: ``torch.FloatTensor``\n            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),\n            representing the unormalised score for each (span, antecedent) pair\n            we considered.\n\n        '
        antecedent_scores = self._antecedent_scorer(self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = Variable(antecedent_scores.data.new(*shape).fill_(0), requires_grad=False)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CoreferenceResolver':
        if False:
            return 10
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        context_layer = Seq2SeqEncoder.from_params(params.pop('context_layer'))
        mention_feedforward = FeedForward.from_params(params.pop('mention_feedforward'))
        antecedent_feedforward = FeedForward.from_params(params.pop('antecedent_feedforward'))
        feature_size = params.pop('feature_size')
        max_span_width = params.pop('max_span_width')
        spans_per_word = params.pop('spans_per_word')
        max_antecedents = params.pop('max_antecedents')
        lexical_dropout = params.pop('lexical_dropout', 0.2)
        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab, text_field_embedder=text_field_embedder, context_layer=context_layer, mention_feedforward=mention_feedforward, antecedent_feedforward=antecedent_feedforward, feature_size=feature_size, max_span_width=max_span_width, spans_per_word=spans_per_word, max_antecedents=max_antecedents, lexical_dropout=lexical_dropout, initializer=initializer, regularizer=regularizer)