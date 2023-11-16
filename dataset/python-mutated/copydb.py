import os
import queue
import sqlalchemy as sa
from twisted.internet import defer
from buildbot import config as config_module
from buildbot.db import connector
from buildbot.db import exceptions
from buildbot.db import model
from buildbot.master import BuildMaster
from buildbot.scripts import base
from buildbot.util import in_reactor
from buildbot.util import misc

@in_reactor
def copy_database(config):
    if False:
        print('Hello World!')
    return _copy_database_in_reactor(config)

@defer.inlineCallbacks
def _copy_database_in_reactor(config):
    if False:
        for i in range(10):
            print('nop')
    if not base.checkBasedir(config):
        return 1
    print_debug = not config['quiet']

    def print_log(*args, **kwargs):
        if False:
            while True:
                i = 10
        if print_debug:
            print(*args, **kwargs)
    config['basedir'] = os.path.abspath(config['basedir'])
    with base.captureErrors((SyntaxError, ImportError), f"Unable to load 'buildbot.tac' from '{config['basedir']}':"):
        config_file = base.getConfigFileFromTac(config['basedir'])
    with base.captureErrors(config_module.ConfigErrors, f"Unable to load '{config_file}' from '{config['basedir']}':"):
        master_src_cfg = base.loadConfig(config, config_file)
        master_dst_cfg = base.loadConfig(config, config_file)
        master_dst_cfg.db['db_url'] = config['destination_url']
    print_log(f"Copying database ({master_src_cfg.db['db_url']}) to ({config['destination_url']})")
    if not master_src_cfg or not master_dst_cfg:
        return 1
    master_src = BuildMaster(config['basedir'])
    master_src.config = master_src_cfg
    try:
        yield master_src.db.setup(check_version=True, verbose=not config['quiet'])
    except exceptions.DatabaseNotReadyError:
        for l in connector.upgrade_message.format(basedir=config['basedir']).split('\n'):
            print(l)
        return 1
    master_dst = BuildMaster(config['basedir'])
    master_dst.config = master_dst_cfg
    yield master_dst.db.setup(check_version=False, verbose=not config['quiet'])
    yield master_dst.db.model.upgrade()
    yield _copy_database_with_db(master_src.db, master_dst.db, print_log)
    return 0

@defer.inlineCallbacks
def _copy_single_table(src_db, dst_db, table, table_name, buildset_to_parent_buildid, print_log):
    if False:
        i = 10
        return i + 15
    column_keys = table.columns.keys()
    rows_queue = queue.Queue(1024)
    written_count = [0]
    total_count = [0]
    autoincrement_foreign_key_column = None
    for (column_name, column) in table.columns.items():
        if not column.foreign_keys and column.primary_key and isinstance(column.type, sa.Integer):
            autoincrement_foreign_key_column = column_name

    def thd_write(conn):
        if False:
            i = 10
            return i + 15
        max_column_id = 0
        while True:
            try:
                rows = rows_queue.get(timeout=1)
                if rows is None:
                    if autoincrement_foreign_key_column is not None and max_column_id != 0:
                        if dst_db.pool.engine.dialect.name == 'postgresql':
                            seq_name = f'{table_name}_{autoincrement_foreign_key_column}_seq'
                            transaction = conn.begin()
                            conn.execute(f'ALTER SEQUENCE {seq_name} RESTART WITH {max_column_id + 1}')
                            transaction.commit()
                    rows_queue.task_done()
                    return
                row_dicts = [{k: getattr(row, k) for k in column_keys} for row in rows]
                if autoincrement_foreign_key_column is not None:
                    for row in row_dicts:
                        max_column_id = max(max_column_id, row[autoincrement_foreign_key_column])
                if table_name == 'buildsets':
                    for row_dict in row_dicts:
                        if row_dict['parent_buildid'] is not None:
                            buildset_to_parent_buildid.append((row_dict['id'], row_dict['parent_buildid']))
                        row_dict['parent_buildid'] = None
            except queue.Empty:
                continue
            try:
                written_count[0] += len(rows)
                print_log(f'Copying {len(rows)} items ({written_count[0]}/{total_count[0]}) for {table_name} table')
                if len(row_dicts) > 0:
                    conn.execute(table.insert(), row_dicts)
            finally:
                rows_queue.task_done()

    def thd_read(conn):
        if False:
            for i in range(10):
                print('nop')
        q = sa.select([sa.sql.func.count()]).select_from(table)
        total_count[0] = conn.execute(q).scalar()
        rows = []
        for row in conn.execute(sa.select(table)).fetchall():
            rows.append(row)
            if len(rows) >= 10000:
                rows_queue.put(rows)
                rows = []
        rows_queue.put(rows)
        rows_queue.put(None)
    yield src_db.pool.do(thd_read)
    yield dst_db.pool.do(thd_write)
    rows_queue.join()

@defer.inlineCallbacks
def _copy_database_with_db(src_db, dst_db, print_log):
    if False:
        i = 10
        return i + 15
    table_names = ['buildsets', 'buildset_properties', 'projects', 'builders', 'changesources', 'buildrequests', 'workers', 'masters', 'buildrequest_claims', 'changesource_masters', 'builder_masters', 'configured_workers', 'connected_workers', 'patches', 'sourcestamps', 'buildset_sourcestamps', 'changes', 'change_files', 'change_properties', 'users', 'users_info', 'change_users', 'builds', 'build_properties', 'build_data', 'steps', 'logs', 'logchunks', 'schedulers', 'scheduler_masters', 'scheduler_changes', 'tags', 'builders_tags', 'test_result_sets', 'test_names', 'test_code_paths', 'test_results', 'objects', 'object_state']
    metadata = model.Model.metadata
    assert len(set(table_names)) == len(set(metadata.tables.keys()))
    buildset_to_parent_buildid = []
    for table_name in table_names:
        table = metadata.tables[table_name]
        yield _copy_single_table(src_db, dst_db, table, table_name, buildset_to_parent_buildid, print_log)

    def thd_write_buildset_parent_buildid(conn):
        if False:
            while True:
                i = 10
        written_count = 0
        for rows in misc.chunkify_list(buildset_to_parent_buildid, 10000):
            q = model.Model.buildsets.update()
            q = q.where(model.Model.buildsets.c.id == sa.bindparam('_id'))
            q = q.values({'parent_buildid': sa.bindparam('parent_buildid')})
            written_count += len(rows)
            print_log(f'Copying {len(rows)} items ({written_count}/{len(buildset_to_parent_buildid)}) for buildset.parent_buildid field')
            conn.execute(q, [{'_id': buildset_id, 'parent_buildid': parent_buildid} for (buildset_id, parent_buildid) in rows])
    yield dst_db.pool.do(thd_write_buildset_parent_buildid)
    print_log('Copy complete')