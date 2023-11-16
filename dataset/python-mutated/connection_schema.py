from __future__ import annotations
import json
from typing import NamedTuple
from marshmallow import Schema, fields
from marshmallow_sqlalchemy import SQLAlchemySchema, auto_field
from airflow.models.connection import Connection

class ConnectionCollectionItemSchema(SQLAlchemySchema):
    """Schema for a connection item."""

    class Meta:
        """Meta."""
        model = Connection
    connection_id = auto_field('conn_id', required=True)
    conn_type = auto_field(required=True)
    description = auto_field()
    host = auto_field()
    login = auto_field()
    schema = auto_field()
    port = auto_field()

class ConnectionSchema(ConnectionCollectionItemSchema):
    """Connection schema."""
    password = auto_field(load_only=True)
    extra = fields.Method('serialize_extra', deserialize='deserialize_extra', allow_none=True)

    @staticmethod
    def serialize_extra(obj: Connection):
        if False:
            return 10
        if obj.extra is None:
            return
        from airflow.utils.log.secrets_masker import redact
        try:
            extra = json.loads(obj.extra)
            return json.dumps(redact(extra))
        except json.JSONDecodeError:
            return obj.extra

    @staticmethod
    def deserialize_extra(value):
        if False:
            return 10
        return value

class ConnectionCollection(NamedTuple):
    """List of Connections with meta."""
    connections: list[Connection]
    total_entries: int

class ConnectionCollectionSchema(Schema):
    """Connection Collection Schema."""
    connections = fields.List(fields.Nested(ConnectionCollectionItemSchema))
    total_entries = fields.Int()

class ConnectionTestSchema(Schema):
    """connection Test Schema."""
    status = fields.Boolean(required=True)
    message = fields.String(required=True)
connection_schema = ConnectionSchema()
connection_collection_item_schema = ConnectionCollectionItemSchema()
connection_collection_schema = ConnectionCollectionSchema()
connection_test_schema = ConnectionTestSchema()