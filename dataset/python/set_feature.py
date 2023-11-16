from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import JACCARD, MODEL_ECD, SET, SIGMOID_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import (
    ecd_defaults_config_registry,
    ecd_input_config_registry,
    ecd_output_config_registry,
    input_mixin_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@input_mixin_registry.register(SET)
@ludwig_dataclass
class SetInputFeatureConfigMixin(BaseMarshmallowConfig):
    """SetInputFeatureConfigMixin is a dataclass that configures the parameters used in both the set input feature
    and the set global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=SET,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(SET)
@ludwig_dataclass
class SetInputFeatureConfig(SetInputFeatureConfigMixin, BaseInputFeatureConfig):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    type: str = schema_utils.ProtectedString(SET)


@DeveloperAPI
@output_mixin_registry.register(SET)
@ludwig_dataclass
class SetOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """SetOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the set output
    feature and the set global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=SET,
        default="classifier",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=SET,
        default=SIGMOID_CROSS_ENTROPY,
    )


@DeveloperAPI
@ecd_output_config_registry.register(SET)
@ludwig_dataclass
class SetOutputFeatureConfig(SetOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    type: str = schema_utils.ProtectedString(SET)

    default_validation_metric: str = schema_utils.StringOptions(
        [JACCARD],
        default=JACCARD,
        description="Internal only use parameter: default validation metric for set output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[SET]["dependencies"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="set_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[SET]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[SET]["reduce_input"],
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Tokens with predicted"
        "probabilities greater than or equal to threshold are predicted to be in the output set (True).",
        parameter_metadata=FEATURE_METADATA[SET]["threshold"],
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(SET)
@ludwig_dataclass
class SetDefaultsConfig(SetInputFeatureConfigMixin, SetOutputFeatureConfigMixin):
    pass
