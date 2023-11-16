"""
 Sequence feature extraction class for common feature extractors to preprocess sequences.
"""
from typing import Dict, List, Optional, Union
import numpy as np
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .utils import PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, logging, to_numpy
logger = logging.get_logger(__name__)

class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (`int`):
            The feature dimension of the extracted features.
        sampling_rate (`int`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.padding_side = kwargs.pop('padding_side', 'right')
        self.return_attention_mask = kwargs.pop('return_attention_mask', True)
        super().__init__(**kwargs)

    def pad(self, processed_features: Union[BatchFeature, List[BatchFeature], Dict[str, BatchFeature], Dict[str, List[BatchFeature]], List[Dict[str, BatchFeature]]], padding: Union[bool, str, PaddingStrategy]=True, max_length: Optional[int]=None, truncation: bool=False, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None) -> BatchFeature:
        if False:
            while True:
                i = 10
        "\n        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the\n        max sequence length in the batch.\n\n        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,\n        `self.padding_value`)\n\n        <Tip>\n\n        If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the\n        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of\n        PyTorch tensors, you will lose the specific device of your tensors however.\n\n        </Tip>\n\n        Args:\n            processed_features ([`BatchFeature`], list of [`BatchFeature`], `Dict[str, List[float]]`, `Dict[str, List[List[float]]` or `List[Dict[str, List[float]]]`):\n                Processed inputs. Can represent one input ([`BatchFeature`] or `Dict[str, List[float]]`) or a batch of\n                input values / vectors (list of [`BatchFeature`], *Dict[str, List[List[float]]]* or *List[Dict[str,\n                List[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader\n                collate function.\n\n                Instead of `List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),\n                see the note above for the return type.\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):\n                Select a strategy to pad the returned sequences (according to the model's padding side and padding\n                index) among:\n\n                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            max_length (`int`, *optional*):\n                Maximum length of the returned list and optionally padding length (see above).\n            truncation (`bool`):\n                Activates truncation to cut input sequences longer than `max_length` to `max_length`.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.\n            return_attention_mask (`bool`, *optional*):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific feature_extractor's default.\n\n                [What are attention masks?](../glossary#attention-mask)\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n        "
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {key: [example[key] for example in processed_features] for key in processed_features[0].keys()}
        if self.model_input_names[0] not in processed_features:
            raise ValueError(f'You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature` to this method that includes {self.model_input_names[0]}, but you provided {list(processed_features.keys())}')
        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        if len(required_input) == 0:
            if return_attention_mask:
                processed_features['attention_mask'] = []
            return processed_features
        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        if return_tensors is None:
            if is_tf_tensor(first_element):
                return_tensors = 'tf'
            elif is_torch_tensor(first_element):
                return_tensors = 'pt'
            elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
                return_tensors = 'np'
            else:
                raise ValueError(f'type of {first_element} unknown: {type(first_element)}. Should be one of a python, numpy, pytorch or tensorflow object.')
        for (key, value) in processed_features.items():
            if isinstance(value[0], (int, float)):
                processed_features[key] = to_numpy(value)
            else:
                processed_features[key] = [to_numpy(v) for v in value]
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)
        required_input = processed_features[self.model_input_names[0]]
        batch_size = len(required_input)
        if not all((len(v) == batch_size for v in processed_features.values())):
            raise ValueError('Some items in the output dictionary have a different batch size than others.')
        truncated_inputs = []
        for i in range(batch_size):
            inputs = {k: v[i] for (k, v) in processed_features.items()}
            inputs_slice = self._truncate(inputs, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, truncation=truncation)
            truncated_inputs.append(inputs_slice)
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max((len(input_slice[self.model_input_names[0]]) for input_slice in truncated_inputs))
            padding_strategy = PaddingStrategy.MAX_LENGTH
        batch_outputs = {}
        for i in range(batch_size):
            outputs = self._pad(truncated_inputs[i], max_length=max_length, padding_strategy=padding_strategy, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
            for (key, value) in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    value = value.astype(np.float32)
                batch_outputs[key].append(value)
        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(self, processed_features: Union[Dict[str, np.ndarray], BatchFeature], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Pad inputs (on left/right and up to predefined length or max length in the batch)\n\n        Args:\n            processed_features (`Union[Dict[str, np.ndarray], BatchFeature]`):\n                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch\n                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)\n            max_length (`int`, *optional*):\n                Maximum length of the returned list and optionally padding length (see below)\n            padding_strategy (`PaddingStrategy`, *optional*, default to `PaddingStrategy.DO_NOT_PAD`):\n                PaddingStrategy to use for padding.\n\n                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch\n                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)\n                - PaddingStrategy.DO_NOT_PAD: Do not pad\n                The feature_extractor padding sides are defined in self.padding_side:\n\n                    - 'left': pads on the left of the sequences\n                    - 'right': pads on the right of the sequences\n            pad_to_multiple_of (`int`, *optional*):\n                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to\n                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs\n                which benefit from having sequence lengths be a multiple of 128.\n            return_attention_mask (`bool`, *optional*):\n                Set to False to avoid returning attention mask (default: set to model specifics)\n        "
        required_input = processed_features[self.model_input_names[0]]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) < max_length
        if return_attention_mask and 'attention_mask' not in processed_features:
            processed_features['attention_mask'] = np.ones(len(required_input), dtype=np.int32)
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == 'right':
                if return_attention_mask:
                    processed_features['attention_mask'] = np.pad(processed_features['attention_mask'], (0, difference))
                padding_shape = ((0, difference), (0, 0)) if self.feature_size > 1 else (0, difference)
                processed_features[self.model_input_names[0]] = np.pad(required_input, padding_shape, 'constant', constant_values=self.padding_value)
            elif self.padding_side == 'left':
                if return_attention_mask:
                    processed_features['attention_mask'] = np.pad(processed_features['attention_mask'], (difference, 0))
                padding_shape = ((difference, 0), (0, 0)) if self.feature_size > 1 else (difference, 0)
                processed_features[self.model_input_names[0]] = np.pad(required_input, padding_shape, 'constant', constant_values=self.padding_value)
            else:
                raise ValueError('Invalid padding strategy:' + str(self.padding_side))
        return processed_features

    def _truncate(self, processed_features: Union[Dict[str, np.ndarray], BatchFeature], max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, truncation: Optional[bool]=None):
        if False:
            return 10
        '\n        Truncate inputs to predefined length or max length in the batch\n\n        Args:\n            processed_features(`Union[Dict[str, np.ndarray], BatchFeature]`):\n                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch\n                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)\n            max_length (`int`, *optional*):\n                maximum length of the returned list and optionally padding length (see below)\n            pad_to_multiple_of (`int`, *optional*) :\n                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to\n                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs\n                which benefit from having sequence lengths be a multiple of 128.\n            truncation (`bool`, *optional*):\n                Activates truncation to cut input sequences longer than `max_length` to `max_length`.\n        '
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError('When setting ``truncation=True``, make sure that ``max_length`` is defined.')
        required_input = processed_features[self.model_input_names[0]]
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_truncated = len(required_input) > max_length
        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            if 'attention_mask' in processed_features:
                processed_features['attention_mask'] = processed_features['attention_mask'][:max_length]
        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None):
        if False:
            print('Hello World!')
        '\n        Find the correct padding strategy\n        '
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(f'When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined')
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and self.padding_value is None:
            raise ValueError('Asking to pad but the feature_extractor does not have a padding value. Please select a value to use as `padding_value`. For example: `feature_extractor.padding_value = 0.0`.')
        return padding_strategy