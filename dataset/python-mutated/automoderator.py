from datetime import timedelta
from pycassa.cassandra.ttypes import NotFoundException
from pycassa.system_manager import ASCII_TYPE, UTF8_TYPE
from r2.lib.db import tdb_cassandra

class PerformedRulesByThing(tdb_cassandra.View):
    """Used to track which rules have previously matched a specific item."""
    _use_db = True
    _connection_pool = 'main'
    _read_consistency_level = tdb_cassandra.CL.QUORUM
    _write_consistency_level = tdb_cassandra.CL.QUORUM
    _ttl = timedelta(days=3)
    _extra_schema_creation_args = {'key_validation_class': ASCII_TYPE, 'column_name_class': ASCII_TYPE, 'default_validation_class': UTF8_TYPE}

    @classmethod
    def _rowkey(cls, thing):
        if False:
            while True:
                i = 10
        return thing._fullname

    @classmethod
    def mark_performed(cls, thing, rule):
        if False:
            return 10
        rowkey = cls._rowkey(thing)
        cls._set_values(rowkey, {rule.unique_id: ''})

    @classmethod
    def get_already_performed(cls, thing):
        if False:
            for i in range(10):
                print('nop')
        rowkey = cls._rowkey(thing)
        try:
            columns = cls._cf.get(rowkey)
        except NotFoundException:
            return []
        return columns.keys()