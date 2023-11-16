from __future__ import annotations
from typing import NamedTuple
from marshmallow import Schema, fields
from marshmallow_sqlalchemy import SQLAlchemySchema, auto_field
from airflow.models.pool import Pool

class PoolSchema(SQLAlchemySchema):
    """Pool schema."""

    class Meta:
        """Meta."""
        model = Pool
    name = auto_field('pool')
    slots = auto_field()
    occupied_slots = fields.Method('get_occupied_slots', dump_only=True)
    running_slots = fields.Method('get_running_slots', dump_only=True)
    queued_slots = fields.Method('get_queued_slots', dump_only=True)
    scheduled_slots = fields.Method('get_scheduled_slots', dump_only=True)
    open_slots = fields.Method('get_open_slots', dump_only=True)
    deferred_slots = fields.Method('get_deferred_slots', dump_only=True)
    description = auto_field()
    include_deferred = fields.Boolean(load_default=False)

    @staticmethod
    def get_occupied_slots(obj: Pool) -> int:
        if False:
            i = 10
            return i + 15
        'Return the occupied slots of the pool.'
        return obj.occupied_slots()

    @staticmethod
    def get_running_slots(obj: Pool) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the running slots of the pool.'
        return obj.running_slots()

    @staticmethod
    def get_queued_slots(obj: Pool) -> int:
        if False:
            i = 10
            return i + 15
        'Return the queued slots of the pool.'
        return obj.queued_slots()

    @staticmethod
    def get_scheduled_slots(obj: Pool) -> int:
        if False:
            print('Hello World!')
        'Return the scheduled slots of the pool.'
        return obj.scheduled_slots()

    @staticmethod
    def get_deferred_slots(obj: Pool) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the deferred slots of the pool.'
        return obj.deferred_slots()

    @staticmethod
    def get_open_slots(obj: Pool) -> float:
        if False:
            while True:
                i = 10
        'Return the open slots of the pool.'
        return obj.open_slots()

class PoolCollection(NamedTuple):
    """List of Pools with metadata."""
    pools: list[Pool]
    total_entries: int

class PoolCollectionSchema(Schema):
    """Pool Collection schema."""
    pools = fields.List(fields.Nested(PoolSchema))
    total_entries = fields.Int()
pool_collection_schema = PoolCollectionSchema()
pool_schema = PoolSchema()