from alembic.migration import MigrationContext
from alembic.operations import Operations
import sqlalchemy as sa
from toolz.curried import do, operator
from zipline.assets.asset_writer import write_version_info
from zipline.utils.compat import wraps
from zipline.errors import AssetDBImpossibleDowngrade
from zipline.utils.preprocess import preprocess
from zipline.utils.sqlite_utils import coerce_string_to_eng

def alter_columns(op, name, *columns, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Alter columns from a table.\n\n    Parameters\n    ----------\n    name : str\n        The name of the table.\n    *columns\n        The new columns to have.\n    selection_string : str, optional\n        The string to use in the selection. If not provided, it will select all\n        of the new columns from the old table.\n\n    Notes\n    -----\n    The columns are passed explicitly because this should only be used in a\n    downgrade where ``zipline.assets.asset_db_schema`` could change.\n    '
    selection_string = kwargs.pop('selection_string', None)
    if kwargs:
        raise TypeError('alter_columns received extra arguments: %r' % sorted(kwargs))
    if selection_string is None:
        selection_string = ', '.join((column.name for column in columns))
    tmp_name = '_alter_columns_' + name
    op.rename_table(name, tmp_name)
    for column in columns:
        for table in (name, tmp_name):
            try:
                op.drop_index('ix_%s_%s' % (table, column.name))
            except sa.exc.OperationalError:
                pass
    op.create_table(name, *columns)
    op.execute('insert into %s select %s from %s' % (name, selection_string, tmp_name))
    op.drop_table(tmp_name)

@preprocess(engine=coerce_string_to_eng(require_exists=True))
def downgrade(engine, desired_version):
    if False:
        i = 10
        return i + 15
    'Downgrades the assets db at the given engine to the desired version.\n\n    Parameters\n    ----------\n    engine : Engine\n        An SQLAlchemy engine to the assets database.\n    desired_version : int\n        The desired resulting version for the assets database.\n    '
    with engine.begin() as conn:
        metadata = sa.MetaData(conn)
        metadata.reflect()
        version_info_table = metadata.tables['version_info']
        starting_version = sa.select((version_info_table.c.version,)).scalar()
        if starting_version < desired_version:
            raise AssetDBImpossibleDowngrade(db_version=starting_version, desired_version=desired_version)
        if starting_version == desired_version:
            return
        ctx = MigrationContext.configure(conn)
        op = Operations(ctx)
        downgrade_keys = range(desired_version, starting_version)[::-1]
        _pragma_foreign_keys(conn, False)
        for downgrade_key in downgrade_keys:
            _downgrade_methods[downgrade_key](op, conn, version_info_table)
        _pragma_foreign_keys(conn, True)

def _pragma_foreign_keys(connection, on):
    if False:
        i = 10
        return i + 15
    'Sets the PRAGMA foreign_keys state of the SQLite database. Disabling\n    the pragma allows for batch modification of tables with foreign keys.\n\n    Parameters\n    ----------\n    connection : Connection\n        A SQLAlchemy connection to the db\n    on : bool\n        If true, PRAGMA foreign_keys will be set to ON. Otherwise, the PRAGMA\n        foreign_keys will be set to OFF.\n    '
    connection.execute('PRAGMA foreign_keys=%s' % ('ON' if on else 'OFF'))
_downgrade_methods = {}

def downgrades(src):
    if False:
        print('Hello World!')
    'Decorator for marking that a method is a downgrade to a version to the\n    previous version.\n\n    Parameters\n    ----------\n    src : int\n        The version this downgrades from.\n\n    Returns\n    -------\n    decorator : callable[(callable) -> callable]\n        The decorator to apply.\n    '

    def _(f):
        if False:
            return 10
        destination = src - 1

        @do(operator.setitem(_downgrade_methods, destination))
        @wraps(f)
        def wrapper(op, conn, version_info_table):
            if False:
                return 10
            conn.execute(version_info_table.delete())
            f(op)
            write_version_info(conn, version_info_table, destination)
        return wrapper
    return _

@downgrades(1)
def _downgrade_v1(op):
    if False:
        i = 10
        return i + 15
    "\n    Downgrade assets db by removing the 'tick_size' column and renaming the\n    'multiplier' column.\n    "
    op.drop_index('ix_futures_contracts_root_symbol')
    op.drop_index('ix_futures_contracts_symbol')
    with op.batch_alter_table('futures_contracts') as batch_op:
        batch_op.alter_column(column_name='multiplier', new_column_name='contract_multiplier')
        batch_op.drop_column('tick_size')
    op.create_index('ix_futures_contracts_root_symbol', table_name='futures_contracts', columns=['root_symbol'])
    op.create_index('ix_futures_contracts_symbol', table_name='futures_contracts', columns=['symbol'], unique=True)

@downgrades(2)
def _downgrade_v2(op):
    if False:
        print('Hello World!')
    "\n    Downgrade assets db by removing the 'auto_close_date' column.\n    "
    op.drop_index('ix_equities_fuzzy_symbol')
    op.drop_index('ix_equities_company_symbol')
    with op.batch_alter_table('equities') as batch_op:
        batch_op.drop_column('auto_close_date')
    op.create_index('ix_equities_fuzzy_symbol', table_name='equities', columns=['fuzzy_symbol'])
    op.create_index('ix_equities_company_symbol', table_name='equities', columns=['company_symbol'])

@downgrades(3)
def _downgrade_v3(op):
    if False:
        while True:
            i = 10
    '\n    Downgrade assets db by adding a not null constraint on\n    ``equities.first_traded``\n    '
    op.create_table('_new_equities', sa.Column('sid', sa.Integer, unique=True, nullable=False, primary_key=True), sa.Column('symbol', sa.Text), sa.Column('company_symbol', sa.Text), sa.Column('share_class_symbol', sa.Text), sa.Column('fuzzy_symbol', sa.Text), sa.Column('asset_name', sa.Text), sa.Column('start_date', sa.Integer, default=0, nullable=False), sa.Column('end_date', sa.Integer, nullable=False), sa.Column('first_traded', sa.Integer, nullable=False), sa.Column('auto_close_date', sa.Integer), sa.Column('exchange', sa.Text))
    op.execute('\n        insert into _new_equities\n        select * from equities\n        where equities.first_traded is not null\n        ')
    op.drop_table('equities')
    op.rename_table('_new_equities', 'equities')
    op.create_index('ix_equities_company_symbol', 'equities', ['company_symbol'])
    op.create_index('ix_equities_fuzzy_symbol', 'equities', ['fuzzy_symbol'])

@downgrades(4)
def _downgrade_v4(op):
    if False:
        while True:
            i = 10
    '\n    Downgrades assets db by copying the `exchange_full` column to `exchange`,\n    then dropping the `exchange_full` column.\n    '
    op.drop_index('ix_equities_fuzzy_symbol')
    op.drop_index('ix_equities_company_symbol')
    op.execute('UPDATE equities SET exchange = exchange_full')
    with op.batch_alter_table('equities') as batch_op:
        batch_op.drop_column('exchange_full')
    op.create_index('ix_equities_fuzzy_symbol', table_name='equities', columns=['fuzzy_symbol'])
    op.create_index('ix_equities_company_symbol', table_name='equities', columns=['company_symbol'])

@downgrades(5)
def _downgrade_v5(op):
    if False:
        for i in range(10):
            print('nop')
    op.create_table('_new_equities', sa.Column('sid', sa.Integer, unique=True, nullable=False, primary_key=True), sa.Column('symbol', sa.Text), sa.Column('company_symbol', sa.Text), sa.Column('share_class_symbol', sa.Text), sa.Column('fuzzy_symbol', sa.Text), sa.Column('asset_name', sa.Text), sa.Column('start_date', sa.Integer, default=0, nullable=False), sa.Column('end_date', sa.Integer, nullable=False), sa.Column('first_traded', sa.Integer), sa.Column('auto_close_date', sa.Integer), sa.Column('exchange', sa.Text), sa.Column('exchange_full', sa.Text))
    op.execute("\n        insert into _new_equities\n        select\n            equities.sid as sid,\n            sym.symbol as symbol,\n            sym.company_symbol as company_symbol,\n            sym.share_class_symbol as share_class_symbol,\n            sym.company_symbol || sym.share_class_symbol as fuzzy_symbol,\n            equities.asset_name as asset_name,\n            equities.start_date as start_date,\n            equities.end_date as end_date,\n            equities.first_traded as first_traded,\n            equities.auto_close_date as auto_close_date,\n            equities.exchange as exchange,\n            equities.exchange_full as exchange_full\n        from\n            equities\n        inner join\n            -- Select the last held symbol for each equity sid from the\n            -- symbol_mappings table. Selecting max(end_date) causes\n            -- SQLite to take the other values from the same row that contained\n            -- the max end_date. See https://www.sqlite.org/lang_select.html#resultset.  # noqa\n            (select\n                 sid, symbol, company_symbol, share_class_symbol, max(end_date)\n             from\n                 equity_symbol_mappings\n             group by sid) as 'sym'\n        on\n            equities.sid == sym.sid\n        ")
    op.drop_table('equity_symbol_mappings')
    op.drop_table('equities')
    op.rename_table('_new_equities', 'equities')
    op.create_index('ix_equities_company_symbol', 'equities', ['company_symbol'])
    op.create_index('ix_equities_fuzzy_symbol', 'equities', ['fuzzy_symbol'])

@downgrades(6)
def _downgrade_v6(op):
    if False:
        return 10
    op.drop_table('equity_supplementary_mappings')

@downgrades(7)
def _downgrade_v7(op):
    if False:
        i = 10
        return i + 15
    tmp_name = '_new_equities'
    op.create_table(tmp_name, sa.Column('sid', sa.Integer, unique=True, nullable=False, primary_key=True), sa.Column('asset_name', sa.Text), sa.Column('start_date', sa.Integer, default=0, nullable=False), sa.Column('end_date', sa.Integer, nullable=False), sa.Column('first_traded', sa.Integer), sa.Column('auto_close_date', sa.Integer), sa.Column('exchange', sa.Text), sa.Column('exchange_full', sa.Text))
    op.execute("\n        insert into\n            _new_equities\n        select\n            eq.sid,\n            eq.asset_name,\n            eq.start_date,\n            eq.end_date,\n            eq.first_traded,\n            eq.auto_close_date,\n            ex.canonical_name,\n            ex.exchange\n        from\n            equities eq\n        inner join\n            exchanges ex\n        on\n            eq.exchange == ex.exchange\n        where\n            ex.country_code in ('US', '??')\n        ")
    op.drop_table('equities')
    op.rename_table(tmp_name, 'equities')
    alter_columns(op, 'futures_root_symbols', sa.Column('root_symbol', sa.Text, unique=True, nullable=False, primary_key=True), sa.Column('root_symbol_id', sa.Integer), sa.Column('sector', sa.Text), sa.Column('description', sa.Text), sa.Column('exchange', sa.Text))
    alter_columns(op, 'futures_contracts', sa.Column('sid', sa.Integer, unique=True, nullable=False, primary_key=True), sa.Column('symbol', sa.Text, unique=True, index=True), sa.Column('root_symbol', sa.Text, index=True), sa.Column('asset_name', sa.Text), sa.Column('start_date', sa.Integer, default=0, nullable=False), sa.Column('end_date', sa.Integer, nullable=False), sa.Column('first_traded', sa.Integer), sa.Column('exchange', sa.Text), sa.Column('notice_date', sa.Integer, nullable=False), sa.Column('expiration_date', sa.Integer, nullable=False), sa.Column('auto_close_date', sa.Integer, nullable=False), sa.Column('multiplier', sa.Float), sa.Column('tick_size', sa.Float))
    alter_columns(op, 'exchanges', sa.Column('exchange', sa.Text, unique=True, nullable=False, primary_key=True), sa.Column('timezone', sa.Text), selection_string='exchange, NULL')
    op.rename_table('exchanges', 'futures_exchanges')
    alter_columns(op, 'futures_root_symbols', sa.Column('root_symbol', sa.Text, unique=True, nullable=False, primary_key=True), sa.Column('root_symbol_id', sa.Integer), sa.Column('sector', sa.Text), sa.Column('description', sa.Text), sa.Column('exchange', sa.Text, sa.ForeignKey('futures_exchanges.exchange')))
    alter_columns(op, 'futures_contracts', sa.Column('sid', sa.Integer, unique=True, nullable=False, primary_key=True), sa.Column('symbol', sa.Text, unique=True, index=True), sa.Column('root_symbol', sa.Text, sa.ForeignKey('futures_root_symbols.root_symbol'), index=True), sa.Column('asset_name', sa.Text), sa.Column('start_date', sa.Integer, default=0, nullable=False), sa.Column('end_date', sa.Integer, nullable=False), sa.Column('first_traded', sa.Integer), sa.Column('exchange', sa.Text, sa.ForeignKey('futures_exchanges.exchange')), sa.Column('notice_date', sa.Integer, nullable=False), sa.Column('expiration_date', sa.Integer, nullable=False), sa.Column('auto_close_date', sa.Integer, nullable=False), sa.Column('multiplier', sa.Float), sa.Column('tick_size', sa.Float))
    op.execute('\n        DELETE FROM\n            equity_symbol_mappings\n        WHERE\n            sid NOT IN (SELECT sid FROM equities);\n        ')
    op.execute('\n        DELETE FROM\n            asset_router\n        WHERE\n            sid\n            NOT IN (\n                SELECT sid FROM equities\n                UNION\n                SELECT sid FROM futures_contracts\n            );\n        ')