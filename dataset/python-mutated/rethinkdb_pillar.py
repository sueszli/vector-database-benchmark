"""
Provide external pillar data from RethinkDB

.. versionadded:: 2018.3.0

:depends: rethinkdb (on the salt-master)


salt master rethinkdb configuration
===================================
These variables must be configured in your master configuration file.
    * ``rethinkdb.host`` - The RethinkDB server. Defaults to ``'salt'``
    * ``rethinkdb.port`` - The port the RethinkDB server listens on.
      Defaults to ``'28015'``
    * ``rethinkdb.database`` - The database to connect to.
      Defaults to ``'salt'``
    * ``rethinkdb.username`` - The username for connecting to RethinkDB.
      Defaults to ``''``
    * ``rethinkdb.password`` - The password for connecting to RethinkDB.
      Defaults to ``''``


salt-master ext_pillar configuration
====================================

The ext_pillar function arguments are given in single line dictionary notation.

.. code-block:: yaml

  ext_pillar:
    - rethinkdb: {table: ext_pillar, id_field: minion_id, field: pillar_root, pillar_key: external_pillar}

In the example above the following happens.
    * The salt-master will look for external pillars in the 'ext_pillar' table
      on the RethinkDB host
    * The minion id will be matched against the 'minion_id' field
    * Pillars will be retrieved from the nested field 'pillar_root'
    * Found pillars will be merged inside a key called 'external_pillar'


Module Documentation
====================
"""
import logging
try:
    import rethinkdb
    HAS_RETHINKDB = True
except ImportError:
    HAS_RETHINKDB = False
__virtualname__ = 'rethinkdb'
__opts__ = {'rethinkdb.host': 'salt', 'rethinkdb.port': '28015', 'rethinkdb.database': 'salt', 'rethinkdb.username': None, 'rethinkdb.password': None}

def __virtual__():
    if False:
        print('Hello World!')
    if not HAS_RETHINKDB:
        return False
    return True
log = logging.getLogger(__name__)

def ext_pillar(minion_id, pillar, table='pillar', id_field=None, field=None, pillar_key=None):
    if False:
        print('Hello World!')
    "\n    Collect minion external pillars from a RethinkDB database\n\n    Arguments:\n\n    * `table`: The RethinkDB table containing external pillar information.\n      Defaults to ``'pillar'``\n    * `id_field`: Field in document containing the minion id.\n      If blank then we assume the table index matches minion ids\n    * `field`: Specific field in the document used for pillar data, if blank\n      then the entire document will be used\n    * `pillar_key`: The salt-master will nest found external pillars under\n      this key before merging into the minion pillars. If blank, external\n      pillars will be merged at top level\n    "
    host = __opts__['rethinkdb.host']
    port = __opts__['rethinkdb.port']
    database = __opts__['rethinkdb.database']
    username = __opts__['rethinkdb.username']
    password = __opts__['rethinkdb.password']
    log.debug("Connecting to %s:%s as user '%s' for RethinkDB ext_pillar", host, port, username)
    conn = rethinkdb.connect(host=host, port=port, db=database, user=username, password=password)
    data = None
    try:
        if id_field:
            log.debug('ext_pillar.rethinkdb: looking up pillar. table: %s, field: %s, minion: %s', table, id_field, minion_id)
            if field:
                data = rethinkdb.table(table).filter({id_field: minion_id}).pluck(field).run(conn)
            else:
                data = rethinkdb.table(table).filter({id_field: minion_id}).run(conn)
        else:
            log.debug('ext_pillar.rethinkdb: looking up pillar. table: %s, field: id, minion: %s', table, minion_id)
            if field:
                data = rethinkdb.table(table).get(minion_id).pluck(field).run(conn)
            else:
                data = rethinkdb.table(table).get(minion_id).run(conn)
    finally:
        if conn.is_open():
            conn.close()
    if data.items:
        if len(data.items) > 1:
            log.error('ext_pillar.rethinkdb: ambiguous documents found for minion %s', minion_id)
            return {}
        else:
            result = data.items.pop()
        if pillar_key:
            return {pillar_key: result}
        return result
    else:
        log.debug('ext_pillar.rethinkdb: no document found')
        return {}