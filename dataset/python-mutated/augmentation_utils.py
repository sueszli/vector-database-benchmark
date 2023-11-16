from typing import Dict, List, Union
from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry
_augmentation_op_registry = Registry()

@DeveloperAPI
def get_augmentation_op_registry() -> Registry:
    if False:
        return 10
    return _augmentation_op_registry

@DeveloperAPI
def register_augmentation_op(name: str, features: Union[str, List[str]]):
    if False:
        while True:
            i = 10
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        if False:
            print('Hello World!')
        for feature in features:
            augmentation_op_registry = get_augmentation_op_registry().get(feature, {})
            augmentation_op_registry[name] = cls
            get_augmentation_op_registry()[feature] = augmentation_op_registry
        return cls
    return wrap

@DeveloperAPI
def get_augmentation_op(feature_type: str, op_name: str):
    if False:
        i = 10
        return i + 15
    return get_augmentation_op_registry()[feature_type][op_name]

class AugmentationPipelines:
    """Container holding augmentation pipelines defined in the model."""

    def __init__(self, augmentation_pipelines: Dict):
        if False:
            for i in range(10):
                print('nop')
        self.augmentation_pipelines = augmentation_pipelines

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.augmentation_pipelines[key]

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return key in self.augmentation_pipelines

    def __len__(self):
        if False:
            return 10
        return len(self.augmentation_pipelines)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.augmentation_pipelines.__iter__()

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        return self.augmentation_pipelines.items()