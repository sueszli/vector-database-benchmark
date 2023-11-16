import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
logger = get_logger(__name__)
TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        scores (`tf.Tensor` of shape `(batch_size, config.vocab_size)`):\n            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam\n            search or log softmax for each vocabulary token when using beam search.\n        cur_len (`int`):\n            The current length of valid input sequence tokens. In the TF implementation, the input_ids' sequence length\n            is the maximum length generate can produce, and we need to know which of its tokens are valid.\n        kwargs (`Dict[str, Any]`, *optional*):\n            Additional logits processor specific kwargs.\n\n    Return:\n        `tf.Tensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.\n"

class TFLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        'TF method for processing logits.'
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')

class TFLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            return 10
        'TF method for warping logits.'
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')

class TFLogitsProcessorList(list):
    """
    This class can be used to create a list of [`TFLogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`TFLogitsProcessor`] to the
    inputs.
    """

    @add_start_docstrings(TF_LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int, **kwargs) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                if not all((arg in kwargs for arg in list(function_args.keys())[2:])):
                    raise ValueError(f'Make sure that all the required parameters: {list(function_args.keys())} for {processor.__class__} are passed to the logits processor.')
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores

class TFTemperatureLogitsWarper(TFLogitsWarper):
    """
    [`TFLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if False:
            print('Hello World!')
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(f'`temperature` has to be a strictly positive float, but is {temperature}')
        self.temperature = temperature

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        scores = scores / self.temperature
        return scores

class TFTopKLogitsWarper(TFLogitsWarper):
    """
    [`TFLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f'`top_k` has to be a strictly positive integer, but is {top_k}')
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        top_k = min(self.top_k, scores.shape[-1])
        indices_to_remove = scores < tf.math.top_k(scores, k=top_k)[0][..., -1:]
        next_scores = tf.where(indices_to_remove, self.filter_value, scores)
        return next_scores

class TFTopPLogitsWarper(TFLogitsWarper):
    """
    [`TFLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        if False:
            i = 10
            return i + 15
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f'`top_p` has to be a float > 0 and < 1, but is {top_p}')
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(f'`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}')
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (topk_scores, topk_indices) = tf.math.top_k(scores, scores.shape[-1])
        mask_scores = tf.fill(scores.shape, self.filter_value)
        cumulative_probs = tf.math.cumsum(stable_softmax(topk_scores, axis=-1), axis=-1)
        score_mask = cumulative_probs < self.top_p
        score_mask = tf.concat((tf.ones([score_mask.shape[0], 1], dtype=tf.bool), score_mask[:, :-1]), axis=-1)
        score_mask = tf.concat((tf.ones([score_mask.shape[0], self.min_tokens_to_keep], dtype=tf.bool), score_mask[:, self.min_tokens_to_keep:]), axis=-1)
        topk_next_scores = tf.where(score_mask, topk_scores, mask_scores)
        scatter_rows = tf.tile(tf.expand_dims(tf.range(topk_indices.shape[0]), axis=-1), [1, topk_indices.shape[-1]])
        scatter_indices = tf.stack((scatter_rows, topk_indices), axis=-1)
        next_scores = tf.scatter_nd(scatter_indices, topk_next_scores, shape=topk_next_scores.shape)
        return next_scores

class TFMinLengthLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if False:
            print('Hello World!')
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f'`min_length` has to be a positive integer, but is {min_length}')
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f'`eos_token_id` has to be a positive integer, but is {eos_token_id}')
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def _apply_eos_token_mask(self, scores: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        eos_token_id_mask = tf.range(scores.shape[-1]) == self.eos_token_id
        scores = tf.where(eos_token_id_mask, float('-inf'), scores)
        return scores

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        scores = tf.cond(tf.less(cur_len, self.min_length), lambda : self._apply_eos_token_mask(scores), lambda : tf.identity(scores))
        return scores

class TFRepetitionPenaltyLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(f'`penalty` has to be a strictly positive float, but is {penalty}')
        self.penalty = penalty

    def _create_score_penalties(self, input_ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        logit_penalties = tf.gather(logits, input_ids, axis=1, batch_dims=1)
        logit_penalties = tf.where(logit_penalties > 0, 1 / self.penalty, logit_penalties)
        logit_penalties = tf.where(logit_penalties < 0, self.penalty, logit_penalties)
        token_penalties = tf.ones(logits.shape)
        batch_size = input_ids.shape[0]
        seq_len = tf.shape(input_ids)[1]
        indexable_prev_input_ids = tf.concat((tf.expand_dims(tf.repeat(tf.range(batch_size), seq_len), axis=-1), tf.expand_dims(tf.reshape(input_ids, [-1]), axis=-1)), axis=1)
        token_penalties = tf.tensor_scatter_nd_update(token_penalties, indices=indexable_prev_input_ids, updates=tf.reshape(logit_penalties, [-1]))
        return token_penalties

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            return 10
        score_penalties = self._create_score_penalties(input_ids[:, :cur_len], scores)
        scores = tf.math.multiply(scores, score_penalties)
        return scores

class TFNoBadWordsLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the tokens of the words
            that should not appear in the generated text, make sure to set `add_prefix_space=True` when initializing
            the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The `add_prefix_space`
            argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours come from
            `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f'`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.')
        if any((not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids)):
            raise ValueError(f'`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.')
        if any((any((not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in bad_word_ids)) for bad_word_ids in bad_words_ids)):
            raise ValueError(f'Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}.')
        self.bad_word_seqs_ids = tf.ragged.constant(bad_words_ids).to_tensor(default_value=-1)
        bad_word_seqs_len = [len(bad_words) for bad_words in bad_words_ids]
        if any((word_len == 0 for word_len in bad_word_seqs_len)):
            raise ValueError(f'Banned words token sequences {bad_words_ids} cannot have an empty list')
        self.bad_word_seqs_len = tf.convert_to_tensor(bad_word_seqs_len, dtype=tf.int32)
        self.seq_forbidden_tokens = tf.convert_to_tensor([bad_words[-1] for bad_words in bad_words_ids])

    def _calc_row_banned_bad_tokens(self, row_input_ids: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')

        def _tokens_match(bad_word_seq_number):
            if False:
                i = 10
                return i + 15

            def _len_one():
                if False:
                    return 10
                return tf.cond(tf.math.equal(self.bad_word_seqs_len[bad_word_seq_number], 1), lambda : tf.ones((), dtype=tf.bool), _len_greater_than_cur_len)

            def _len_greater_than_cur_len():
                if False:
                    while True:
                        i = 10
                return tf.cond(tf.math.greater(self.bad_word_seqs_len[bad_word_seq_number], tf.shape(row_input_ids)[0]), lambda : tf.zeros((), dtype=tf.bool), _match_found)

            def _match_found():
                if False:
                    while True:
                        i = 10
                compare_len = self.bad_word_seqs_len[bad_word_seq_number] - 1
                return tf.cond(tf.math.reduce_all(tf.math.equal(row_input_ids[-compare_len:], self.bad_word_seqs_ids[bad_word_seq_number, :compare_len])), lambda : tf.ones((), dtype=tf.bool), lambda : tf.zeros((), dtype=tf.bool))
            match = _len_one()
            return match
        match_mask = tf.map_fn(_tokens_match, tf.range(self.bad_word_seqs_ids.shape[0]), fn_output_signature=tf.bool)
        row_banned_tokens = self.seq_forbidden_tokens[match_mask]
        return row_banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')

        def _get_row_updated_score(row_inputs: Tuple[tf.Tensor]) -> tf.Tensor:
            if False:
                while True:
                    i = 10
            (row_input_ids, row_score) = row_inputs
            banned_tokens = self._calc_row_banned_bad_tokens(row_input_ids[:cur_len])
            banned_tokens_mask = tf.scatter_nd(indices=tf.expand_dims(banned_tokens, axis=-1), updates=tf.ones_like(banned_tokens, dtype=tf.bool), shape=row_score.shape)
            row_score = tf.where(banned_tokens_mask, -float('inf'), row_score)
            return row_score
        scores = tf.map_fn(_get_row_updated_score, (input_ids, scores), fn_output_signature=tf.float32)
        return scores

class TFNoRepeatNGramLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if False:
            print('Hello World!')
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f'`ngram_size` has to be a strictly positive integer, but is {ngram_size}')
        self.ngram_size = ngram_size

    def calc_banned_ngram_tokens(self, input_ids, num_hypos, cur_len):
        if False:
            i = 10
            return i + 15
        if cur_len + 1 < self.ngram_size:
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        prev_input_ids = input_ids[:, :cur_len]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].numpy().tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            if False:
                print('Hello World!')
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])
        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        if not tf.executing_eagerly():
            raise NotImplementedError('TFNoRepeatNGramLogitsProcessor is only implemented for eager execution.')
        (batch_size, vocab_size) = scores.shape
        banned_tokens = self.calc_banned_ngram_tokens(input_ids, batch_size, cur_len)
        banned_tokens_indices_mask = []
        for banned_tokens_slice in banned_tokens:
            banned_tokens_indices_mask.append([True if token in banned_tokens_slice else False for token in range(vocab_size)])
        scores = tf.where(tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float('inf'), scores)
        return scores

class TFForcedBOSTokenLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        if False:
            print('Hello World!')
        if bos_token_id < 0:
            raise ValueError(f'The forced bos token id  must be a non-negative integer, got {bos_token_id}')
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if cur_len == 1:
            (batch_size, num_tokens) = scores.shape
            scores = tf.zeros((batch_size, 1))
            if self.bos_token_id > 0:
                scores = tf.concat((tf.broadcast_to(-float('inf'), (batch_size, self.bos_token_id)), scores), axis=-1)
            if self.bos_token_id < num_tokens - 1:
                scores = tf.concat((scores, tf.broadcast_to(-float('inf'), (batch_size, num_tokens - 1 - self.bos_token_id))), axis=-1)
        return scores

class TFForcedEOSTokenLogitsProcessor(TFLogitsProcessor):
    """
    [`TFLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        if False:
            i = 10
            return i + 15
        self.max_length = max_length
        if eos_token_id < 0:
            raise ValueError(f'The forced eos token id must be a non-negative integer, got {eos_token_id}')
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if cur_len == self.max_length - 1:
            (batch_size, num_tokens) = scores.shape
            scores = tf.zeros((batch_size, 1))
            if self.eos_token_id > 0:
                scores = tf.concat((tf.broadcast_to(-float('inf'), (batch_size, self.eos_token_id)), scores), axis=-1)
            if self.eos_token_id < num_tokens - 1:
                scores = tf.concat((scores, tf.broadcast_to(-float('inf'), (batch_size, num_tokens - 1 - self.eos_token_id))), axis=-1)
        return scores

class TFSuppressTokensAtBeginLogitsProcessor(TFLogitsProcessor):
    """
    [`TFSuppressTokensAtBeginLogitsProcessor`] suppresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        if False:
            return 10
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            print('Hello World!')
        scores = tf.cond(tf.equal(cur_len, self.begin_index), lambda : tf.tensor_scatter_nd_update(scores, indices=[[i, token] for i in range(scores.shape[0]) for token in self.begin_suppress_tokens], updates=[-float('inf') for _ in range(scores.shape[0] * len(self.begin_suppress_tokens))]), lambda : scores)
        return scores

class TFSuppressTokensLogitsProcessor(TFLogitsProcessor):
    """This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they
    are not sampled."""

    def __init__(self, suppress_tokens):
        if False:
            return 10
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        scores = tf.tensor_scatter_nd_update(scores, indices=[[i, token] for i in range(scores.shape[0]) for token in self.suppress_tokens], updates=[-float('inf') for _ in range(scores.shape[0] * len(self.suppress_tokens))])
        return scores

class TFForceTokensLogitsProcessor(TFLogitsProcessor):
    """This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `0` and all other tokens to
    `-inf` so that they are sampled at their corresponding index."""

    def __init__(self, force_token_map: List[List[int]]):
        if False:
            i = 10
            return i + 15
        force_token_map = dict(force_token_map)
        force_token_array = np.ones(max(force_token_map.keys()) + 1, dtype=np.int32) * -1
        for (index, token) in force_token_map.items():
            if token is not None:
                force_token_array[index] = token
        self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)

    def __call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')

        def _force_token(generation_idx):
            if False:
                return 10
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]
            new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float('inf')
            indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
            updates = tf.zeros((batch_size,), dtype=scores.dtype)
            new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
            return new_scores
        scores = tf.cond(tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]), lambda : tf.identity(scores), lambda : tf.cond(tf.greater_equal(self.force_token_array[cur_len], 0), lambda : _force_token(cur_len), lambda : scores))
        return scores