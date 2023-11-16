"""
Speech processor class for Wav2Vec2
"""
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends
logger = logging.get_logger(__name__)
if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC
    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils import PreTrainedTokenizerBase
ListOfDict = List[Dict[str, Union[int, str]]]

@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2DecoderWithLM`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        logit_score (list of `float` or `float`):
            Total logit score of the beams associated with produced text.
        lm_score (list of `float`):
            Fused lm_score of the beams associated with produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """
    text: Union[List[List[str]], List[str], str]
    logit_score: Union[List[List[float]], List[float], float] = None
    lm_score: Union[List[List[float]], List[float], float] = None
    word_offsets: Union[List[List[ListOfDict]], List[ListOfDict], ListOfDict] = None

class Wav2Vec2ProcessorWithLM(ProcessorMixin):
    """
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
    with language model support into a single processor for language model boosted speech recognition decoding.

    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`]):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`Wav2Vec2CTCTokenizer`]):
            An instance of [`Wav2Vec2CTCTokenizer`]. The tokenizer is a required input.
        decoder (`pyctcdecode.BeamSearchDecoderCTC`):
            An instance of [`pyctcdecode.BeamSearchDecoderCTC`]. The decoder is a required input.
    """
    feature_extractor_class = 'Wav2Vec2FeatureExtractor'
    tokenizer_class = 'Wav2Vec2CTCTokenizer'

    def __init__(self, feature_extractor: 'FeatureExtractionMixin', tokenizer: 'PreTrainedTokenizerBase', decoder: 'BeamSearchDecoderCTC'):
        if False:
            return 10
        from pyctcdecode import BeamSearchDecoderCTC
        super().__init__(feature_extractor, tokenizer)
        if not isinstance(decoder, BeamSearchDecoderCTC):
            raise ValueError(f'`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}')
        missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(f"The tokens {missing_decoder_tokens} are defined in the tokenizer's vocabulary, but not in the decoder's alphabet. Make sure to include {missing_decoder_tokens} in the decoder's alphabet.")
        self.decoder = decoder
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def save_pretrained(self, save_directory):
        if False:
            while True:
                i = 10
        super().save_pretrained(save_directory)
        self.decoder.save_to_dir(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Instantiate a [`Wav2Vec2ProcessorWithLM`] from a pretrained Wav2Vec2 processor.\n\n        <Tip>\n\n        This class method is simply calling Wav2Vec2FeatureExtractor's\n        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], Wav2Vec2CTCTokenizer's\n        [`~tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`], and\n        [`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`].\n\n        Please refer to the docstrings of the methods above for more information.\n\n        </Tip>\n\n        Args:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                This can be either:\n\n                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on\n                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or\n                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.\n                - a path to a *directory* containing a feature extractor file saved using the\n                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.\n                - a path or url to a saved feature extractor JSON *file*, e.g.,\n                  `./my_model_directory/preprocessor_config.json`.\n            **kwargs\n                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and\n                [`PreTrainedTokenizer`]\n        "
        requires_backends(cls, 'pyctcdecode')
        from pyctcdecode import BeamSearchDecoderCTC
        (feature_extractor, tokenizer) = super()._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path):
            decoder = BeamSearchDecoderCTC.load_from_dir(pretrained_model_name_or_path)
        else:
            kwargs.pop('_from_auto', None)
            kwargs.pop('trust_remote_code', None)
            language_model_filenames = os.path.join(BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, '*')
            alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
            allow_patterns = [language_model_filenames, alphabet_filename]
            decoder = BeamSearchDecoderCTC.load_from_hf_hub(pretrained_model_name_or_path, allow_patterns=allow_patterns, **kwargs)
        for attribute in ['alpha', 'beta', 'unk_score_offset', 'score_boundary']:
            value = kwargs.pop(attribute, None)
            if value is not None:
                cls._set_language_model_attribute(decoder, attribute, value)
        missing_decoder_tokens = cls.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(f"The tokens {missing_decoder_tokens} are defined in the tokenizer's vocabulary, but not in the decoder's alphabet. Make sure to include {missing_decoder_tokens} in the decoder's alphabet.")
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=decoder)

    @staticmethod
    def _set_language_model_attribute(decoder: 'BeamSearchDecoderCTC', attribute: str, value: float):
        if False:
            i = 10
            return i + 15
        setattr(decoder.model_container[decoder._model_key], attribute, value)

    @property
    def language_model(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder.model_container[self.decoder._model_key]

    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())
        for (i, token) in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ''
            if token == tokenizer.word_delimiter_token:
                tokenizer_vocab_list[i] = ' '
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN
        missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)
        return missing_tokens

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's\n        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output. If used in the context\n        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to\n        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.__call__`]. Please refer to the docstring of the above two\n        methods for more information.\n        "
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)
        if 'raw_speech' in kwargs:
            warnings.warn('Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.')
            audio = kwargs.pop('raw_speech')
        else:
            audio = kwargs.pop('audio', None)
        sampling_rate = kwargs.pop('sampling_rate', None)
        text = kwargs.pop('text', None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]
        if audio is None and text is None:
            raise ValueError('You need to specify either an `audio` or `text` input to process.')
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)
        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs['labels'] = encodings['input_ids']
            return inputs

    def pad(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's\n        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context\n        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to\n        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.pad`]. Please refer to the docstring of the above two methods\n        for more information.\n        "
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)
        input_features = kwargs.pop('input_features', None)
        labels = kwargs.pop('labels', None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features['labels'] = labels['input_ids']
            return input_features

    def batch_decode(self, logits: np.ndarray, pool: Optional[Pool]=None, num_processes: Optional[int]=None, beam_width: Optional[int]=None, beam_prune_logp: Optional[float]=None, token_min_logp: Optional[float]=None, hotwords: Optional[Iterable[str]]=None, hotword_weight: Optional[float]=None, alpha: Optional[float]=None, beta: Optional[float]=None, unk_score_offset: Optional[float]=None, lm_score_boundary: Optional[bool]=None, output_word_offsets: bool=False, n_best: int=1):
        if False:
            print('Hello World!')
        "\n        Batch decode output logits to audio transcription with language model support.\n\n        <Tip>\n\n        This function makes use of Python's multiprocessing. Currently, multiprocessing is available only on Unix\n        systems (see this [issue](https://github.com/kensho-technologies/pyctcdecode/issues/65)).\n\n        If you are decoding multiple batches, consider creating a `Pool` and passing it to `batch_decode`. Otherwise,\n        `batch_decode` will be very slow since it will create a fresh `Pool` for each call. See usage example below.\n\n        </Tip>\n\n        Args:\n            logits (`np.ndarray`):\n                The logits output vector of the model representing the log probabilities for each token.\n            pool (`multiprocessing.Pool`, *optional*):\n                An optional user-managed pool. If not set, one will be automatically created and closed. The pool\n                should be instantiated *after* `Wav2Vec2ProcessorWithLM`. Otherwise, the LM won't be available to the\n                pool's sub-processes.\n\n                <Tip>\n\n                Currently, only pools created with a 'fork' context can be used. If a 'spawn' pool is passed, it will\n                be ignored and sequential decoding will be used instead.\n\n                </Tip>\n\n            num_processes (`int`, *optional*):\n                If `pool` is not set, number of processes on which the function should be parallelized over. Defaults\n                to the number of available CPUs.\n            beam_width (`int`, *optional*):\n                Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.\n            beam_prune_logp (`int`, *optional*):\n                Beams that are much worse than best beam will be pruned Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.\n            token_min_logp (`int`, *optional*):\n                Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode's\n                DEFAULT_MIN_TOKEN_LOGP.\n            hotwords (`List[str]`, *optional*):\n                List of words with extra importance, can be OOV for LM\n            hotword_weight (`int`, *optional*):\n                Weight factor for hotword importance Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.\n            alpha (`float`, *optional*):\n                Weight for language model during shallow fusion\n            beta (`float`, *optional*):\n                Weight for length score adjustment of during scoring\n            unk_score_offset (`float`, *optional*):\n                Amount of log score offset for unknown tokens\n            lm_score_boundary (`bool`, *optional*):\n                Whether to have kenlm respect boundaries when scoring\n            output_word_offsets (`bool`, *optional*, defaults to `False`):\n                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate\n                and model downsampling rate to compute the time-stamps of transcribed words.\n            n_best (`int`, *optional*, defaults to `1`):\n                Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list\n                of lists of strings, `logit_score` will be a list of lists of floats, and `lm_score` will be a list of\n                lists of floats, where the length of the outer list will correspond to the batch size and the length of\n                the inner list will correspond to the number of returned hypotheses . The value should be >= 1.\n\n                <Tip>\n\n                Please take a look at the Example of [`~Wav2Vec2ProcessorWithLM.decode`] to better understand how to\n                make use of `output_word_offsets`. [`~Wav2Vec2ProcessorWithLM.batch_decode`] works the same way with\n                batched output.\n\n                </Tip>\n\n        Returns:\n            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`].\n\n        Example:\n            See [Decoding multiple audios](#decoding-multiple-audios).\n        "
        from pyctcdecode.constants import DEFAULT_BEAM_WIDTH, DEFAULT_HOTWORD_WEIGHT, DEFAULT_MIN_TOKEN_LOGP, DEFAULT_PRUNE_LOGP
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT
        self.decoder.reset_params(alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary)
        logits_list = [array[(array != -100.0).all(axis=-1)] for array in logits]
        if pool is None:
            default_context = get_start_method()
            if default_context == 'fork':
                cm = pool = get_context().Pool(num_processes)
            else:
                logger.warning('Parallel batch decoding is not currently supported in this platform. Falling back to sequential decoding.')
                cm = nullcontext()
        else:
            cm = nullcontext()
            if num_processes is not None:
                logger.warning('Parameter `num_process` was passed, but it will be ignored since `pool` was also specified.')
        with cm:
            decoded_beams = self.decoder.decode_beams_batch(pool=pool, logits_list=logits_list, beam_width=beam_width, beam_prune_logp=beam_prune_logp, token_min_logp=token_min_logp, hotwords=hotwords, hotword_weight=hotword_weight)
        (batch_texts, logit_scores, lm_scores, word_offsets) = ([], [], [], [])
        for d in decoded_beams:
            batch_texts.append([beam[0] for beam in d])
            logit_scores.append([beam[-2] for beam in d])
            lm_scores.append([beam[-1] for beam in d])
            word_offsets.append([[{'word': word, 'start_offset': start_offset, 'end_offset': end_offset} for (word, (start_offset, end_offset)) in beam[1]] for beam in d])
        word_offsets = word_offsets if output_word_offsets else None
        if n_best == 1:
            return Wav2Vec2DecoderWithLMOutput(text=[hyps[0] for hyps in batch_texts], logit_score=[hyps[0] for hyps in logit_scores], lm_score=[hyps[0] for hyps in lm_scores], word_offsets=[hyps[0] for hyps in word_offsets] if word_offsets is not None else None)
        else:
            return Wav2Vec2DecoderWithLMOutput(text=[hyps[:n_best] for hyps in batch_texts], logit_score=[hyps[:n_best] for hyps in logit_scores], lm_score=[hyps[:n_best] for hyps in lm_scores], word_offsets=[hyps[:n_best] for hyps in word_offsets] if word_offsets is not None else None)

    def decode(self, logits: np.ndarray, beam_width: Optional[int]=None, beam_prune_logp: Optional[float]=None, token_min_logp: Optional[float]=None, hotwords: Optional[Iterable[str]]=None, hotword_weight: Optional[float]=None, alpha: Optional[float]=None, beta: Optional[float]=None, unk_score_offset: Optional[float]=None, lm_score_boundary: Optional[bool]=None, output_word_offsets: bool=False, n_best: int=1):
        if False:
            return 10
        '\n        Decode output logits to audio transcription with language model support.\n\n        Args:\n            logits (`np.ndarray`):\n                The logits output vector of the model representing the log probabilities for each token.\n            beam_width (`int`, *optional*):\n                Maximum number of beams at each step in decoding. Defaults to pyctcdecode\'s DEFAULT_BEAM_WIDTH.\n            beam_prune_logp (`int`, *optional*):\n                A threshold to prune beams with log-probs less than best_beam_logp + beam_prune_logp. The value should\n                be <= 0. Defaults to pyctcdecode\'s DEFAULT_PRUNE_LOGP.\n            token_min_logp (`int`, *optional*):\n                Tokens with log-probs below token_min_logp are skipped unless they are have the maximum log-prob for an\n                utterance. Defaults to pyctcdecode\'s DEFAULT_MIN_TOKEN_LOGP.\n            hotwords (`List[str]`, *optional*):\n                List of words with extra importance which can be missing from the LM\'s vocabulary, e.g. ["huggingface"]\n            hotword_weight (`int`, *optional*):\n                Weight multiplier that boosts hotword scores. Defaults to pyctcdecode\'s DEFAULT_HOTWORD_WEIGHT.\n            alpha (`float`, *optional*):\n                Weight for language model during shallow fusion\n            beta (`float`, *optional*):\n                Weight for length score adjustment of during scoring\n            unk_score_offset (`float`, *optional*):\n                Amount of log score offset for unknown tokens\n            lm_score_boundary (`bool`, *optional*):\n                Whether to have kenlm respect boundaries when scoring\n            output_word_offsets (`bool`, *optional*, defaults to `False`):\n                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate\n                and model downsampling rate to compute the time-stamps of transcribed words.\n            n_best (`int`, *optional*, defaults to `1`):\n                Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list\n                of strings, `logit_score` will be a list of floats, and `lm_score` will be a list of floats, where the\n                length of these lists will correspond to the number of returned hypotheses. The value should be >= 1.\n\n                <Tip>\n\n                Please take a look at the example below to better understand how to make use of `output_word_offsets`.\n\n                </Tip>\n\n        Returns:\n            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`].\n\n        Example:\n\n        ```python\n        >>> # Let\'s see how to retrieve time steps for a model\n        >>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC\n        >>> from datasets import load_dataset\n        >>> import datasets\n        >>> import torch\n\n        >>> # import model, feature extractor, tokenizer\n        >>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")\n        >>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")\n\n        >>> # load first sample of English common_voice\n        >>> dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)\n        >>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))\n        >>> dataset_iter = iter(dataset)\n        >>> sample = next(dataset_iter)\n\n        >>> # forward sample through model to get greedily predicted transcription ids\n        >>> input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values\n        >>> with torch.no_grad():\n        ...     logits = model(input_values).logits[0].cpu().numpy()\n\n        >>> # retrieve word stamps (analogous commands for `output_char_offsets`)\n        >>> outputs = processor.decode(logits, output_word_offsets=True)\n        >>> # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate\n        >>> time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate\n\n        >>> word_offsets = [\n        ...     {\n        ...         "word": d["word"],\n        ...         "start_time": round(d["start_offset"] * time_offset, 2),\n        ...         "end_time": round(d["end_offset"] * time_offset, 2),\n        ...     }\n        ...     for d in outputs.word_offsets\n        ... ]\n        >>> # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:\n        >>> # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en\n        >>> word_offsets[:4]\n        [{\'word\': \'THE\', \'start_time\': 0.68, \'end_time\': 0.78}, {\'word\': \'TRACK\', \'start_time\': 0.88, \'end_time\': 1.1}, {\'word\': \'APPEARS\', \'start_time\': 1.18, \'end_time\': 1.66}, {\'word\': \'ON\', \'start_time\': 1.86, \'end_time\': 1.92}]\n        ```'
        from pyctcdecode.constants import DEFAULT_BEAM_WIDTH, DEFAULT_HOTWORD_WEIGHT, DEFAULT_MIN_TOKEN_LOGP, DEFAULT_PRUNE_LOGP
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT
        self.decoder.reset_params(alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary)
        decoded_beams = self.decoder.decode_beams(logits, beam_width=beam_width, beam_prune_logp=beam_prune_logp, token_min_logp=token_min_logp, hotwords=hotwords, hotword_weight=hotword_weight)
        word_offsets = None
        if output_word_offsets:
            word_offsets = [[{'word': word, 'start_offset': start_offset, 'end_offset': end_offset} for (word, (start_offset, end_offset)) in beam[2]] for beam in decoded_beams]
        logit_scores = [beam[-2] for beam in decoded_beams]
        lm_scores = [beam[-1] for beam in decoded_beams]
        hypotheses = [beam[0] for beam in decoded_beams]
        if n_best > len(decoded_beams):
            logger.info('N-best size is larger than the number of generated hypotheses, all hypotheses will be returned.')
        if n_best == 1:
            return Wav2Vec2DecoderWithLMOutput(text=hypotheses[0], logit_score=logit_scores[0], lm_score=lm_scores[0], word_offsets=word_offsets[0] if word_offsets is not None else None)
        else:
            return Wav2Vec2DecoderWithLMOutput(text=hypotheses[:n_best], logit_score=logit_scores[:n_best], lm_score=lm_scores[:n_best], word_offsets=word_offsets[:n_best] if word_offsets is not None else None)

    @contextmanager
    def as_target_processor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Temporarily sets the processor for processing the target. Useful for encoding the labels when fine-tuning\n        Wav2Vec2.\n        '
        warnings.warn('`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your labels by using the argument `text` of the regular `__call__` method (either in the same call as your audio inputs, or in a separate call.')
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False