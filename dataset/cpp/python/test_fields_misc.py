from typing import Dict, Tuple, Union

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass

from ludwig.config_validation.validation import get_validator, validate
from ludwig.schema import utils as schema_utils


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


# Simple marshmallow fields:


def test_StringOptions():
    # Test case of default conflicting with allowed options:
    test_options = ["one"]
    with pytest.raises(AssertionError):
        schema_utils.StringOptions(test_options, default=None, allow_none=False)

    # Test creating a schema with simple option, null not allowed:
    test_options = ["one"]

    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: str = schema_utils.StringOptions(test_options, "one", allow_none=False)

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": None})


# Complex, custom marshmallow fields:


def test_Embed():
    # Test metadata matches expected defaults after field creation (null allowed):
    default_embed = get_marshmallow_from_dataclass_field(schema_utils.Embed())
    assert default_embed.default is None
    assert default_embed.allow_none is True

    # Test simple schema creation:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[None, str, int] = schema_utils.Embed()

    # Test null/empty loading cases:
    assert CustomTestSchema.Schema().load({}).foo is None
    assert CustomTestSchema.Schema().load({"foo": None}).foo is None

    # Test valid strings/numbers:
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "not_add"})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {}})

    assert CustomTestSchema.Schema().load({"foo": "add"}).foo == "add"
    assert CustomTestSchema.Schema().load({"foo": 1}).foo == 1


def test_InitializerOrDict():
    # Test metadata matches expected defaults after field creation (null allowed):
    default_initializerordict = get_marshmallow_from_dataclass_field(schema_utils.InitializerOrDict())
    assert default_initializerordict.default == "xavier_uniform"

    initializerordict = get_marshmallow_from_dataclass_field(schema_utils.InitializerOrDict("zeros"))
    assert initializerordict.default == "zeros"

    # Test default value validation:
    with pytest.raises(MarshmallowValidationError):
        schema_utils.InitializerOrDict("test")

    # Test simple schema creation:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[None, str, Dict] = schema_utils.InitializerOrDict()

    # Test invalid non-dict loads:
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": 1})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    # Test valid loads:
    assert CustomTestSchema.Schema().load({}).foo == "xavier_uniform"
    assert CustomTestSchema.Schema().load({"foo": "zeros"}).foo == "zeros"

    # Test invalid dict loads:
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": None})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"a": "b"}})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"type": "invalid"}})

    # Test valid dict loads:
    assert CustomTestSchema.Schema().load({"foo": {"type": "zeros"}}).foo == {"type": "zeros"}


def test_FloatRangeTupleDataclassField():
    # Test metadata matches expected defaults after field creation (null not allowed):
    default_floatrange_tuple = get_marshmallow_from_dataclass_field(schema_utils.FloatRangeTupleDataclassField())
    assert default_floatrange_tuple.default == (0.9, 0.999)

    # Test dimensional mismatch:
    with pytest.raises(MarshmallowValidationError):
        schema_utils.FloatRangeTupleDataclassField(n=3, default=(1, 1))

    # Test default schema creation:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(allow_none=True)

    # Test empty load:
    assert CustomTestSchema.Schema().load({}).foo == (0.9, 0.999)
    assert CustomTestSchema.Schema().load({"foo": None}).foo is None

    # Test invalid loads (null, non-float values, wrong dimension):
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": [1, "test"]})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": (1, 1, 1)})

    # Test non-default schema (N=3, other custom metadata):
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(n=3, default=(1, 1, 1), min=-10, max=10)

    assert CustomTestSchema.Schema().load({}).foo == (1, 1, 1)
    assert CustomTestSchema.Schema().load({"foo": [2, 2, 2]}).foo == (2, 2, 2)
    assert CustomTestSchema.Schema().load({"foo": (2, 2, 2)}).foo == (2, 2, 2)


def test_OneOfOptionsField():
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[float, str] = schema_utils.OneOfOptionsField(
            default=0.1,
            description="",
            allow_none=False,
            field_options=[
                schema_utils.FloatRange(default=0.001, min=0, max=1, allow_none=False),
                schema_utils.StringOptions(options=["placeholder"], default="placeholder", allow_none=False),
            ],
        )

    # Test valid loads:
    assert CustomTestSchema.Schema().load({}).foo == 0.1
    CustomTestSchema().foo == 0.1

    # Test invalid loads:
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": None})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    # Reverse the order and allow none (via StringOptions):
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[None, float, str] = schema_utils.OneOfOptionsField(
            default="placeholder",
            description="",
            field_options=[
                schema_utils.FloatRange(default=0.001, min=0, max=1, allow_none=False),
                schema_utils.StringOptions(options=["placeholder"], default="placeholder", allow_none=False),
            ],
            allow_none=True,
        )

    # Test valid loads:
    assert CustomTestSchema.Schema().load({}).foo == "placeholder"
    assert CustomTestSchema.Schema().load({"foo": 0.1}).foo == 0.1
    CustomTestSchema().foo == "placeholder"
    CustomTestSchema.Schema().load({"foo": None})

    # Test title generation:
    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema)
    assert json["properties"]["foo"]["title"] == "foo"
    assert json["properties"]["foo"]["oneOf"][0]["title"] == "foo_float_option"

    # Test invalid loads:
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "bar"})


def test_OneOfOptionsField_allows_none():
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[float, str] = schema_utils.OneOfOptionsField(
            default=None,
            allow_none=True,
            description="",
            field_options=[
                schema_utils.PositiveInteger(description="", default=1, allow_none=False),
                schema_utils.List(list_type=int, allow_none=False),
            ],
        )

    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema)
    schema = {
        "type": "object",
        "properties": {
            "hello": json,
        },
        "definitions": {},
    }
    validate(instance={"hello": {"foo": None}}, schema=schema, cls=get_validator())


def test_OneOfOptionsField_allows_none_fails_if_multiple_fields_allow_none():
    with pytest.raises(ValueError):

        @dataclass
        class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
            foo: Union[float, str] = schema_utils.OneOfOptionsField(
                default=None,
                description="",
                field_options=[
                    schema_utils.PositiveInteger(description="", default=1, allow_none=True),
                    schema_utils.List(list_type=int, allow_none=True),
                ],
            )


def test_OneOfOptionsField_allows_none_one_field_allows_none():
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Union[float, str] = schema_utils.OneOfOptionsField(
            default=None,
            description="",
            field_options=[
                schema_utils.PositiveInteger(description="", default=1, allow_none=False),
                schema_utils.List(list_type=int, allow_none=True),
            ],
        )

    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema)
    schema = {
        "type": "object",
        "properties": {
            "hello": json,
        },
        "definitions": {},
    }
    validate(instance={"hello": {"foo": None}}, schema=schema, cls=get_validator())
