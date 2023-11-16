import json
from datetime import datetime
from pycassa.system_manager import UTF8_TYPE, TIME_UUID_TYPE
from pycassa.util import convert_uuid_to_time
from pylons import app_globals as g
from r2.lib.db import tdb_cassandra

class AdminNotesBySystem(tdb_cassandra.View):
    _use_db = True
    _read_consistency_level = tdb_cassandra.CL.QUORUM
    _write_consistency_level = tdb_cassandra.CL.ONE
    _compare_with = TIME_UUID_TYPE
    _extra_schema_creation_args = {'key_validation_class': UTF8_TYPE, 'default_validation_class': UTF8_TYPE}

    @classmethod
    def add(cls, system_name, subject, note, author, when=None):
        if False:
            for i in range(10):
                print('nop')
        if not when:
            when = datetime.now(g.tz)
        jsonpacked = json.dumps({'note': note, 'author': author})
        updatedict = {when: jsonpacked}
        key = cls._rowkey(system_name, subject)
        cls._set_values(key, updatedict)

    @classmethod
    def in_display_order(cls, system_name, subject):
        if False:
            for i in range(10):
                print('nop')
        key = cls._rowkey(system_name, subject)
        try:
            query = cls._cf.get(key, column_reversed=True)
        except tdb_cassandra.NotFoundException:
            return []
        result = []
        for (uuid, json_blob) in query.iteritems():
            when = datetime.fromtimestamp(convert_uuid_to_time(uuid), tz=g.tz)
            payload = json.loads(json_blob)
            payload['when'] = when
            result.append(payload)
        return result

    @classmethod
    def _rowkey(cls, system_name, subject):
        if False:
            i = 10
            return i + 15
        return '%s:%s' % (system_name, subject)