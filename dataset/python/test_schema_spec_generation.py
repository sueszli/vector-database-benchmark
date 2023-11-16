from typing import Dict, List, Optional

import attrs

from litestar import post
from litestar.testing import create_test_client
from tests.models import DataclassPet


def test_spec_generation() -> None:
    @attrs.define
    class Person:
        first_name: str
        last_name: str
        id: str
        optional: Optional[str]
        complex: Dict[str, List[Dict[str, str]]]
        pets: Optional[List[DataclassPet]]

    @post("/")
    def handler(data: Person) -> Person:
        return data

    with create_test_client(handler) as client:
        schema = client.app.openapi_schema
        assert schema

        assert schema.to_schema()["components"]["schemas"]["Person"] == {
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "id": {"type": "string"},
                "optional": {"oneOf": [{"type": "null"}, {"type": "string"}]},
                "complex": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                },
                "pets": {
                    "oneOf": [
                        {"type": "null"},
                        {"items": {"$ref": "#/components/schemas/DataclassPet"}, "type": "array"},
                    ]
                },
            },
            "type": "object",
            "required": ["complex", "first_name", "id", "last_name"],
            "title": "Person",
        }
