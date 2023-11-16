import pytest

from ludwig.config_validation.validation import check_schema
from ludwig.constants import TRAINER
from ludwig.error import ConfigValidationError
from ludwig.schema.optimizers import optimizer_registry
from ludwig.schema.trainer import ECDTrainerConfig
from tests.integration_tests.utils import binary_feature, category_feature, number_feature

# Note: simple tests for now, but once we add dependent fields we can add tests for more complex relationships in this
# file. Currently verifies that the nested fields work, as the others are covered by basic marshmallow validation:


def test_config_trainer_empty_null_and_default():
    config = {
        "input_features": [
            category_feature(encoder={"type": "dense", "vocab_size": 2}, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature()],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    check_schema(config)

    config[TRAINER] = None
    with pytest.raises(ConfigValidationError):
        check_schema(config)

    config[TRAINER] = ECDTrainerConfig.Schema().dump({})
    check_schema(config)


def test_config_trainer_bad_optimizer():
    config = {
        "input_features": [
            category_feature(encoder={"type": "dense", "vocab_size": 2}, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature()],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    check_schema(config)

    # Test manually set-to-null optimizer vs unspecified:
    config[TRAINER]["optimizer"] = None
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    assert ECDTrainerConfig.Schema().load({}).optimizer is not None

    # Test all types in optimizer_registry supported:
    for key in optimizer_registry.keys():
        config[TRAINER]["optimizer"] = {"type": key}
        check_schema(config)

    # Test invalid optimizer type:
    config[TRAINER]["optimizer"] = {"type": 0}
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]["optimizer"] = {"type": "invalid"}
    with pytest.raises(ConfigValidationError):
        check_schema(config)


def test_optimizer_property_validation():
    config = {
        "input_features": [
            category_feature(encoder={"type": "dense", "vocab_size": 2}, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature()],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    check_schema(config)

    # Test that an optimizer's property types are enforced:
    config[TRAINER]["optimizer"] = {"type": "rmsprop"}
    check_schema(config)

    config[TRAINER]["optimizer"]["momentum"] = "invalid"
    with pytest.raises(ConfigValidationError):
        check_schema(config)

    # Test extra keys are excluded and defaults are loaded appropriately:
    config[TRAINER]["optimizer"]["momentum"] = 10
    config[TRAINER]["optimizer"]["extra_key"] = "invalid"
    check_schema(config)
    assert not hasattr(ECDTrainerConfig.Schema().load(config[TRAINER]).optimizer, "extra_key")

    # Test bad parameter range:
    config[TRAINER]["optimizer"] = {"type": "rmsprop", "eps": -1}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

    # Test config validation for tuple types:
    config[TRAINER]["optimizer"] = {"type": "adam", "betas": (0.1, 0.1)}
    check_schema(config)


def test_clipper_property_validation():
    config = {
        "input_features": [
            category_feature(encoder={"type": "dense", "vocab_size": 2}, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature()],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    check_schema(config)

    # Test null/empty clipper:
    config[TRAINER]["gradient_clipping"] = None
    check_schema(config)
    config[TRAINER]["gradient_clipping"] = {}
    check_schema(config)
    assert (
        ECDTrainerConfig.Schema().load(config[TRAINER]).gradient_clipping
        == ECDTrainerConfig.Schema().load({}).gradient_clipping
    )

    # Test invalid clipper type:
    config[TRAINER]["gradient_clipping"] = 0
    with pytest.raises(ConfigValidationError):
        check_schema(config)
    config[TRAINER]["gradient_clipping"] = "invalid"
    with pytest.raises(ConfigValidationError):
        check_schema(config)

    # Test that an optimizer's property types are enforced:
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": None}
    check_schema(config)
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": 1}
    check_schema(config)
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": "invalid"}
    with pytest.raises(ConfigValidationError):
        check_schema(config)

    # Test extra keys are excluded and defaults are loaded appropriately:
    config[TRAINER]["gradient_clipping"] = {"clipnorm": 1}
    config[TRAINER]["gradient_clipping"]["extra_key"] = "invalid"
    assert not hasattr(ECDTrainerConfig.Schema().load(config[TRAINER]).gradient_clipping, "extra_key")
