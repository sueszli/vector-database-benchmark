import inspect
from typing import List, Union
import numpy as np
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, ChunkPipeline
logger = logging.get_logger(__name__)

class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if False:
            i = 10
            return i + 15
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(',') if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if False:
            i = 10
            return i + 15
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError('You must include at least one label and at least one sequence.')
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError('The provided hypothesis_template "{}" was not able to be formatted with the target labels. Make sure the passed template includes formatting syntax such as {{}} where the label should go.'.format(hypothesis_template))
        if isinstance(sequences, str):
            sequences = [sequences]
        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])
        return (sequence_pairs, sequences)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(ChunkPipeline):
    """
    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
    config's :attr:*~transformers.PretrainedConfig.label2id*.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="facebook/bart-large-mnli")
    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["english", "german"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).
    """

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        if False:
            return 10
        self._args_parser = args_parser
        super().__init__(*args, **kwargs)
        if self.entailment_id == -1:
            logger.warning("Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to -1. Define a descriptive label2id mapping in the model config to ensure correct outputs.")

    @property
    def entailment_id(self):
        if False:
            for i in range(10):
                print('nop')
        for (label, ind) in self.model.config.label2id.items():
            if label.lower().startswith('entail'):
                return ind
        return -1

    def _parse_and_tokenize(self, sequence_pairs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.ONLY_FIRST, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated\n        '
        return_tensors = self.framework
        if self.tokenizer.pad_token is None:
            logger.error('Tokenizer was not supporting padding necessary for zero-shot, attempting to use  `pad_token=eos_token`')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(sequence_pairs, add_special_tokens=add_special_tokens, return_tensors=return_tensors, padding=padding, truncation=truncation)
        except Exception as e:
            if 'too short' in str(e):
                inputs = self.tokenizer(sequence_pairs, add_special_tokens=add_special_tokens, return_tensors=return_tensors, padding=padding, truncation=TruncationStrategy.DO_NOT_TRUNCATE)
            else:
                raise e
        return inputs

    def _sanitize_parameters(self, **kwargs):
        if False:
            while True:
                i = 10
        if kwargs.get('multi_class', None) is not None:
            kwargs['multi_label'] = kwargs['multi_class']
            logger.warning('The `multi_class` argument has been deprecated and renamed to `multi_label`. `multi_class` will be removed in a future version of Transformers.')
        preprocess_params = {}
        if 'candidate_labels' in kwargs:
            preprocess_params['candidate_labels'] = self._args_parser._parse_labels(kwargs['candidate_labels'])
        if 'hypothesis_template' in kwargs:
            preprocess_params['hypothesis_template'] = kwargs['hypothesis_template']
        postprocess_params = {}
        if 'multi_label' in kwargs:
            postprocess_params['multi_label'] = kwargs['multi_label']
        return (preprocess_params, {}, postprocess_params)

    def __call__(self, sequences: Union[str, List[str]], *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Classify the sequence(s) given as inputs. See the [`ZeroShotClassificationPipeline`] documentation for more\n        information.\n\n        Args:\n            sequences (`str` or `List[str]`):\n                The sequence(s) to classify, will be truncated if the model input is too large.\n            candidate_labels (`str` or `List[str]`):\n                The set of possible class labels to classify each sequence into. Can be a single label, a string of\n                comma-separated labels, or a list of labels.\n            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):\n                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or\n                similar syntax for the candidate label to be inserted into the template. For example, the default\n                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the\n                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`. The default template\n                works well in many cases, but it may be worthwhile to experiment with different templates depending on\n                the task setting.\n            multi_label (`bool`, *optional*, defaults to `False`):\n                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that\n                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered\n                independent and probabilities are normalized for each candidate by doing a softmax of the entailment\n                score vs. the contradiction score.\n\n        Return:\n            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:\n\n            - **sequence** (`str`) -- The sequence for which this is the output.\n            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.\n            - **scores** (`List[float]`) -- The probabilities for each of the labels.\n        '
        if len(args) == 0:
            pass
        elif len(args) == 1 and 'candidate_labels' not in kwargs:
            kwargs['candidate_labels'] = args[0]
        else:
            raise ValueError(f'Unable to understand extra arguments {args}')
        return super().__call__(sequences, **kwargs)

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template='This example is {}.'):
        if False:
            while True:
                i = 10
        (sequence_pairs, sequences) = self._args_parser(inputs, candidate_labels, hypothesis_template)
        for (i, (candidate_label, sequence_pair)) in enumerate(zip(candidate_labels, sequence_pairs)):
            model_input = self._parse_and_tokenize([sequence_pair])
            yield {'candidate_label': candidate_label, 'sequence': sequences[0], 'is_last': i == len(candidate_labels) - 1, **model_input}

    def _forward(self, inputs):
        if False:
            while True:
                i = 10
        candidate_label = inputs['candidate_label']
        sequence = inputs['sequence']
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        model_forward = self.model.forward if self.framework == 'pt' else self.model.call
        if 'use_cache' in inspect.signature(model_forward).parameters.keys():
            model_inputs['use_cache'] = False
        outputs = self.model(**model_inputs)
        model_outputs = {'candidate_label': candidate_label, 'sequence': sequence, 'is_last': inputs['is_last'], **outputs}
        return model_outputs

    def postprocess(self, model_outputs, multi_label=False):
        if False:
            return 10
        candidate_labels = [outputs['candidate_label'] for outputs in model_outputs]
        sequences = [outputs['sequence'] for outputs in model_outputs]
        logits = np.concatenate([output['logits'].numpy() for output in model_outputs])
        N = logits.shape[0]
        n = len(candidate_labels)
        num_sequences = N // n
        reshaped_outputs = logits.reshape((num_sequences, n, -1))
        if multi_label or len(candidate_labels) == 1:
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]
        else:
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        top_inds = list(reversed(scores[0].argsort()))
        return {'sequence': sequences[0], 'labels': [candidate_labels[i] for i in top_inds], 'scores': scores[0, top_inds].tolist()}