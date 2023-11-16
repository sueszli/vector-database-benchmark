from typing import Any, Dict, TYPE_CHECKING
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, BAG, BINARY, CATEGORY, CATEGORY_DISTRIBUTION, DATE, H3, IMAGE, NUMBER, SEQUENCE, SET, TEXT, TIMESERIES, VECTOR
from ludwig.features.audio_feature import AudioFeatureMixin, AudioInputFeature
from ludwig.features.bag_feature import BagFeatureMixin, BagInputFeature
from ludwig.features.binary_feature import BinaryFeatureMixin, BinaryInputFeature, BinaryOutputFeature
from ludwig.features.category_feature import CategoryDistributionFeatureMixin, CategoryDistributionOutputFeature, CategoryFeatureMixin, CategoryInputFeature, CategoryOutputFeature
from ludwig.features.date_feature import DateFeatureMixin, DateInputFeature
from ludwig.features.h3_feature import H3FeatureMixin, H3InputFeature
from ludwig.features.image_feature import ImageFeatureMixin, ImageInputFeature
from ludwig.features.number_feature import NumberFeatureMixin, NumberInputFeature, NumberOutputFeature
from ludwig.features.sequence_feature import SequenceFeatureMixin, SequenceInputFeature, SequenceOutputFeature
from ludwig.features.set_feature import SetFeatureMixin, SetInputFeature, SetOutputFeature
from ludwig.features.text_feature import TextFeatureMixin, TextInputFeature, TextOutputFeature
from ludwig.features.timeseries_feature import TimeseriesFeatureMixin, TimeseriesInputFeature, TimeseriesOutputFeature
from ludwig.features.vector_feature import VectorFeatureMixin, VectorInputFeature, VectorOutputFeature
from ludwig.utils.misc_utils import get_from_registry
if TYPE_CHECKING:
    from ludwig.models.base import BaseModel
    from ludwig.schema.model_types.base import ModelConfig

@DeveloperAPI
def get_base_type_registry() -> Dict:
    if False:
        i = 10
        return i + 15
    return {TEXT: TextFeatureMixin, CATEGORY: CategoryFeatureMixin, SET: SetFeatureMixin, BAG: BagFeatureMixin, BINARY: BinaryFeatureMixin, NUMBER: NumberFeatureMixin, SEQUENCE: SequenceFeatureMixin, TIMESERIES: TimeseriesFeatureMixin, IMAGE: ImageFeatureMixin, AUDIO: AudioFeatureMixin, H3: H3FeatureMixin, DATE: DateFeatureMixin, VECTOR: VectorFeatureMixin, CATEGORY_DISTRIBUTION: CategoryDistributionFeatureMixin}

@DeveloperAPI
def get_input_type_registry() -> Dict:
    if False:
        i = 10
        return i + 15
    return {TEXT: TextInputFeature, NUMBER: NumberInputFeature, BINARY: BinaryInputFeature, CATEGORY: CategoryInputFeature, SET: SetInputFeature, SEQUENCE: SequenceInputFeature, IMAGE: ImageInputFeature, AUDIO: AudioInputFeature, TIMESERIES: TimeseriesInputFeature, BAG: BagInputFeature, H3: H3InputFeature, DATE: DateInputFeature, VECTOR: VectorInputFeature}

@DeveloperAPI
def get_output_type_registry() -> Dict:
    if False:
        i = 10
        return i + 15
    return {CATEGORY: CategoryOutputFeature, BINARY: BinaryOutputFeature, NUMBER: NumberOutputFeature, SEQUENCE: SequenceOutputFeature, SET: SetOutputFeature, TEXT: TextOutputFeature, TIMESERIES: TimeseriesOutputFeature, VECTOR: VectorOutputFeature, CATEGORY_DISTRIBUTION: CategoryDistributionOutputFeature}

def update_config_with_metadata(config_obj: 'ModelConfig', training_set_metadata: Dict[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    for input_feature in config_obj.input_features:
        feature = get_from_registry(input_feature.type, get_input_type_registry())
        feature.update_config_with_metadata(input_feature, training_set_metadata[input_feature.name])
    for output_feature in config_obj.output_features:
        feature = get_from_registry(output_feature.type, get_output_type_registry())
        feature.update_config_with_metadata(output_feature, training_set_metadata[output_feature.name])

def update_config_with_model(config_obj: 'ModelConfig', model: 'BaseModel'):
    if False:
        i = 10
        return i + 15
    'Updates the config with the final input feature params given a model.\n\n    This function should only be called to update the config after the model is initialized. Currently only implemented\n    for input features because it is only relevant for HuggingFace text encoders. HuggingFace text encoders only know\n    their final config after class initialization.\n    '
    for input_feature in config_obj.input_features:
        model_input_feature = model.input_features.get(input_feature.name)
        model_input_feature.update_config_after_module_init(input_feature)