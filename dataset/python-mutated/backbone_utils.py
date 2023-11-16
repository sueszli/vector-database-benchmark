""" Collection of utils to be used by backbones and their components."""
import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union

class BackboneType(enum.Enum):
    TIMM = 'timm'
    TRANSFORMERS = 'transformers'

def verify_out_features_out_indices(out_features: Optional[Iterable[str]], out_indices: Optional[Iterable[int]], stage_names: Optional[Iterable[str]]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that out_indices and out_features are valid for the given stage_names.\n    '
    if stage_names is None:
        raise ValueError('Stage_names must be set for transformers backbones')
    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f'out_features must be a list {type(out_features)}')
        if any((feat not in stage_names for feat in out_features)):
            raise ValueError(f'out_features must be a subset of stage_names: {stage_names} got {out_features}')
    if out_indices is not None:
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(f'out_indices must be a list or tuple, got {type(out_indices)}')
        if any((idx >= len(stage_names) for idx in out_indices)):
            raise ValueError('out_indices must be valid indices for stage_names {stage_names}, got {out_indices}')
    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError('out_features and out_indices should have the same length if both are set')
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError('out_features and out_indices should correspond to the same stages if both are set')

def _align_output_features_output_indices(out_features: Optional[List[str]], out_indices: Optional[Union[List[int], Tuple[int]]], stage_names: List[str]):
    if False:
        i = 10
        return i + 15
    '\n    Finds the corresponding `out_features` and `out_indices` for the given `stage_names`.\n\n    The logic is as follows:\n        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the\n        `out_indices`.\n        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the\n        `out_features`.\n        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.\n        - `out_indices` and `out_features` set: input `out_indices` and `out_features` are returned.\n\n    Args:\n        out_features (`List[str]`): The names of the features for the backbone to output.\n        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.\n        stage_names (`List[str]`): The names of the stages of the backbone.\n    '
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return (out_features, out_indices)

def get_aligned_output_features_output_indices(out_features: Optional[List[str]], out_indices: Optional[Union[List[int], Tuple[int]]], stage_names: List[str]) -> Tuple[List[str], List[int]]:
    if False:
        while True:
            i = 10
    '\n    Get the `out_features` and `out_indices` so that they are aligned.\n\n    The logic is as follows:\n        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the\n        `out_indices`.\n        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the\n        `out_features`.\n        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.\n        - `out_indices` and `out_features` set: they are verified to be aligned.\n\n    Args:\n        out_features (`List[str]`): The names of the features for the backbone to output.\n        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.\n        stage_names (`List[str]`): The names of the stages of the backbone.\n    '
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    (output_features, output_indices) = _align_output_features_output_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    return (output_features, output_indices)

class BackboneMixin:
    backbone_type: Optional[BackboneType] = None

    def _init_timm_backbone(self, config) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the backbone model from timm The backbone must already be loaded to self._backbone\n        '
        if getattr(self, '_backbone', None) is None:
            raise ValueError('self._backbone must be set before calling _init_timm_backbone')
        self.stage_names = [stage['module'] for stage in self._backbone.feature_info.info]
        self.num_features = [stage['num_chs'] for stage in self._backbone.feature_info.info]
        out_indices = self._backbone.feature_info.out_indices
        out_features = self._backbone.feature_info.module_name()
        verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=self.stage_names)
        (self._out_features, self._out_indices) = (out_features, out_indices)

    def _init_transformers_backbone(self, config) -> None:
        if False:
            while True:
                i = 10
        stage_names = getattr(config, 'stage_names')
        out_features = getattr(config, 'out_features', None)
        out_indices = getattr(config, 'out_indices', None)
        self.stage_names = stage_names
        (self._out_features, self._out_indices) = get_aligned_output_features_output_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
        self.num_features = None

    def _init_backbone(self, config) -> None:
        if False:
            return 10
        '\n        Method to initialize the backbone. This method is called by the constructor of the base class after the\n        pretrained model weights have been loaded.\n        '
        self.config = config
        self.use_timm_backbone = getattr(config, 'use_timm_backbone', False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS
        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(config)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f'backbone_type {self.backbone_type} not supported.')

    @property
    def out_features(self):
        if False:
            return 10
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        if False:
            print('Hello World!')
        '\n        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.\n        '
        (self._out_features, self._out_indices) = get_aligned_output_features_output_indices(out_features=out_features, out_indices=None, stage_names=self.stage_names)

    @property
    def out_indices(self):
        if False:
            i = 10
            return i + 15
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        if False:
            while True:
                i = 10
        '\n        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.\n        '
        (self._out_features, self._out_indices) = get_aligned_output_features_output_indices(out_features=None, out_indices=out_indices, stage_names=self.stage_names)

    @property
    def out_feature_channels(self):
        if False:
            while True:
                i = 10
        return {stage: self.num_features[i] for (i, stage) in enumerate(self.stage_names)}

    @property
    def channels(self):
        if False:
            while True:
                i = 10
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for (k, v) in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)

    def forward(self, pixel_values, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This method should be implemented by the derived class.')

    def to_dict(self):
        if False:
            print('Hello World!')
        '\n        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to\n        include the `out_features` and `out_indices` attributes.\n        '
        output = super().to_dict()
        output['out_features'] = output.pop('_out_features')
        output['out_indices'] = output.pop('_out_indices')
        return output

class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    @property
    def out_features(self):
        if False:
            i = 10
            return i + 15
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        if False:
            return 10
        '\n        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.\n        '
        (self._out_features, self._out_indices) = get_aligned_output_features_output_indices(out_features=out_features, out_indices=None, stage_names=self.stage_names)

    @property
    def out_indices(self):
        if False:
            i = 10
            return i + 15
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        if False:
            print('Hello World!')
        '\n        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.\n        '
        (self._out_features, self._out_indices) = get_aligned_output_features_output_indices(out_features=None, out_indices=out_indices, stage_names=self.stage_names)

    def to_dict(self):
        if False:
            while True:
                i = 10
        '\n        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to\n        include the `out_features` and `out_indices` attributes.\n        '
        output = super().to_dict()
        output['out_features'] = output.pop('_out_features')
        output['out_indices'] = output.pop('_out_indices')
        return output