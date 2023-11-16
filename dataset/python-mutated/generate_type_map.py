import argparse
import asyncio
import asyncpg
_BUILTIN_ARRAYS = ('_text', '_oid')
_INVALIDOID = 0
_MAXBUILTINOID = 10000 - 1
_TYPE_ALIASES = {'smallint': 'int2', 'int': 'int4', 'integer': 'int4', 'bigint': 'int8', 'decimal': 'numeric', 'real': 'float4', 'double precision': 'float8', 'timestamp with timezone': 'timestamptz', 'timestamp without timezone': 'timestamp', 'time with timezone': 'timetz', 'time without timezone': 'time', 'char': 'bpchar', 'character': 'bpchar', 'character varying': 'varchar', 'bit varying': 'varbit'}

async def runner(args):
    conn = await asyncpg.connect(host=args.pghost, port=args.pgport, user=args.pguser)
    buf = '# Copyright (C) 2016-present the asyncpg authors and contributors\n# <see AUTHORS file>\n#\n# This module is part of asyncpg and is released under\n# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0\n\n\n# GENERATED FROM pg_catalog.pg_type\n' + '# DO NOT MODIFY, use tools/generate_type_map.py to update\n\n' + 'DEF INVALIDOID = {}\n'.format(_INVALIDOID) + 'DEF MAXBUILTINOID = {}\n'.format(_MAXBUILTINOID)
    pg_types = await conn.fetch("\n        SELECT\n            oid,\n            typname\n        FROM\n            pg_catalog.pg_type\n        WHERE\n            typtype IN ('b', 'p')\n            AND (typelem = 0 OR typname = any($1) OR typlen > 0)\n            AND oid <= $2\n        ORDER BY\n            oid\n    ", _BUILTIN_ARRAYS, _MAXBUILTINOID)
    defs = []
    typemap = {}
    array_types = []
    for pg_type in pg_types:
        typeoid = pg_type['oid']
        typename = pg_type['typname']
        defname = '{}OID'.format(typename.upper())
        defs.append('DEF {name} = {oid}'.format(name=defname, oid=typeoid))
        if typename in _BUILTIN_ARRAYS:
            array_types.append(defname)
            typename = typename[1:] + '[]'
        typemap[defname] = typename
    buf += 'DEF MAXSUPPORTEDOID = {}\n\n'.format(pg_types[-1]['oid'])
    buf += '\n'.join(defs)
    buf += '\n\ncdef ARRAY_TYPES = ({},)'.format(', '.join(array_types))
    f_typemap = ('{}: {!r}'.format(dn, n) for (dn, n) in sorted(typemap.items()))
    buf += '\n\nBUILTIN_TYPE_OID_MAP = {{\n    {}\n}}'.format(',\n    '.join(f_typemap))
    buf += '\n\nBUILTIN_TYPE_NAME_MAP = ' + '{v: k for k, v in BUILTIN_TYPE_OID_MAP.items()}'
    for (k, v) in _TYPE_ALIASES.items():
        buf += '\n\nBUILTIN_TYPE_NAME_MAP[{!r}] = \\\n    BUILTIN_TYPE_NAME_MAP[{!r}]'.format(k, v)
    print(buf)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='generate protocol/pgtypes.pxi from pg_catalog.pg_types')
    parser.add_argument('--pghost', type=str, default='127.0.0.1', help='PostgreSQL server host')
    parser.add_argument('--pgport', type=int, default=5432, help='PostgreSQL server port')
    parser.add_argument('--pguser', type=str, default='postgres', help='PostgreSQL server user')
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(runner(args))
if __name__ == '__main__':
    main()