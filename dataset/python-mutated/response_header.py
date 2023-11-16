from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from litestar.exceptions import ImproperlyConfiguredException
if TYPE_CHECKING:
    from litestar.openapi.spec import Example
__all__ = ('ResponseHeader',)

@dataclass
class ResponseHeader:
    """Container type for a response header."""
    name: str
    'Header name'
    documentation_only: bool = False
    'Defines the ResponseHeader instance as for OpenAPI documentation purpose only.'
    value: str | None = None
    'Value to set for the response header.'
    description: str | None = None
    'A brief description of the parameter. This could contain examples of\n    use.\n\n    [CommonMark syntax](https://spec.commonmark.org/) MAY be used for\n    rich text representation.\n    '
    required: bool = False
    'Determines whether this parameter is mandatory.\n\n    If the [parameter location](https://spec.openapis.org/oas/v3.1.0#parameterIn) is `"path"`, this property is **REQUIRED** and its value MUST be `true`.\n    Otherwise, the property MAY be included and its default value is `false`.\n    '
    deprecated: bool = False
    'Specifies that a parameter is deprecated and SHOULD be transitioned out\n    of usage.\n\n    Default value is `false`.\n    '
    allow_empty_value: bool = False
    'Sets the ability to pass empty-valued parameters. This is valid only for\n    `query` parameters and allows sending a parameter with an empty value.\n    Default value is `false`. If.\n\n    [style](https://spec.openapis.org/oas/v3.1.0#parameterStyle) is used, and if behavior is `n/a` (cannot be\n    serialized), the value of `allowEmptyValue` SHALL be ignored. Use of this property is NOT RECOMMENDED, as it is\n    likely to be removed in a later revision.\n\n    The rules for serialization of the parameter are specified in one of two ways.\n    For simpler scenarios, a [schema](https://spec.openapis.org/oas/v3.1.0#parameterSchema) and [style](https://spec.openapis.org/oas/v3.1.0#parameterStyle)\n    can describe the structure and syntax of the parameter.\n    '
    style: str | None = None
    'Describes how the parameter value will be serialized depending on the\n    type of the parameter value. Default values (based on value of `in`):\n\n    - for `query` - `form`;\n    - for `path` - `simple`;\n    - for `header` - `simple`;\n    - for `cookie` - `form`.\n    '
    explode: bool | None = None
    'When this is true, parameter values of type `array` or `object` generate\n    separate parameters for each value of the array or key-value pair of the\n    map.\n\n    For other types of parameters this property has no effect.\n    When [style](https://spec.openapis.org/oas/v3.1.0#parameterStyle) is `form`, the default value is `true`.\n    For all other styles, the default value is `false`.\n    '
    allow_reserved: bool = False
    "Determines whether the parameter value SHOULD allow reserved characters,\n    as defined by.\n\n    [RFC3986](https://tools.ietf.org/html/rfc3986#section-2.2) `:/?#[]@!$&'()*+,;=` to be included without percent-\n    encoding.\n\n    This property only applies to parameters with an `in` value of `query`. The default value is `false`.\n    "
    example: Any | None = None
    "Example of the parameter's potential value.\n\n    The example SHOULD match the specified schema and encoding\n    properties if present. The `example` field is mutually exclusive of\n    the `examples` field. Furthermore, if referencing a `schema` that\n    contains an example, the `example` value SHALL _override_ the\n    example provided by the schema. To represent examples of media types\n    that cannot naturally be represented in JSON or YAML, a string value\n    can contain the example with escaping where necessary.\n    "
    examples: dict[str, Example] | None = None
    "Examples of the parameter's potential value. Each example SHOULD contain\n    a value in the correct format as specified in the parameter encoding. The\n    `examples` field is mutually exclusive of the `example` field. Furthermore,\n    if referencing a `schema` that contains an example, the `examples` value\n    SHALL _override_ the example provided by the schema.\n\n    For more complex scenarios, the [content](https://spec.openapis.org/oas/v3.1.0#parameterContent) property\n    can define the media type and schema of the parameter.\n    A parameter MUST contain either a `schema` property, or a `content` property, but not both.\n    When `example` or `examples` are provided in conjunction with the `schema` object,\n    the example MUST follow the prescribed serialization strategy for the parameter.\n    "

    def __post_init__(self) -> None:
        if False:
            while True:
                i = 10
        'Ensure that either value is set or the instance is for documentation_only.'
        if not self.documentation_only and self.value is None:
            raise ImproperlyConfiguredException('value must be set if documentation_only is false')

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(self.name)