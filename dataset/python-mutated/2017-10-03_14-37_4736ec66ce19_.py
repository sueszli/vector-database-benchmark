"""empty message

Revision ID: 4736ec66ce19
Revises: f959a6652acd
Create Date: 2017-10-03 14:37:01.376578

"""
import logging
import sqlalchemy as sa
from alembic import op
from superset.utils.core import generic_find_fk_constraint_name, generic_find_fk_constraint_names, generic_find_uq_constraint_name
revision = '4736ec66ce19'
down_revision = 'f959a6652acd'
conv = {'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s', 'uq': 'uq_%(table_name)s_%(column_0_name)s'}
datasources = sa.Table('datasources', sa.MetaData(), sa.Column('id', sa.Integer, primary_key=True), sa.Column('datasource_name', sa.String(255)))

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('datasources', naming_convention=conv) as batch_op:
        batch_op.create_unique_constraint('uq_datasources_cluster_name', ['cluster_name', 'datasource_name'])
    for foreign in ['columns', 'metrics']:
        with op.batch_alter_table(foreign, naming_convention=conv) as batch_op:
            batch_op.add_column(sa.Column('datasource_id', sa.Integer))
            batch_op.create_foreign_key(f'fk_{foreign}_datasource_id_datasources', 'datasources', ['datasource_id'], ['id'])
        table = sa.Table(foreign, sa.MetaData(), sa.Column('id', sa.Integer, primary_key=True), sa.Column('datasource_name', sa.String(255)), sa.Column('datasource_id', sa.Integer))
        for datasource in bind.execute(datasources.select()):
            bind.execute(table.update().where(table.c.datasource_name == datasource.datasource_name).values(datasource_id=datasource.id))
        with op.batch_alter_table(foreign, naming_convention=conv) as batch_op:
            names = generic_find_fk_constraint_names(foreign, {'datasource_name'}, 'datasources', insp)
            for name in names:
                batch_op.drop_constraint(name or f'fk_{foreign}_datasource_name_datasources', type_='foreignkey')
            batch_op.drop_column('datasource_name')
    try:
        with op.batch_alter_table('datasources', naming_convention=conv) as batch_op:
            batch_op.drop_constraint(generic_find_uq_constraint_name('datasources', {'datasource_name'}, insp) or 'uq_datasources_datasource_name', type_='unique')
    except Exception as ex:
        logging.warning('Constraint drop failed, you may want to do this manually on your database. For context, this is a known issue around nondeterministic constraint names on Postgres and perhaps more databases through SQLAlchemy.')
        logging.exception(ex)

def downgrade():
    if False:
        return 10
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('datasources', naming_convention=conv) as batch_op:
        batch_op.create_unique_constraint('uq_datasources_datasource_name', ['datasource_name'])
    for foreign in ['columns', 'metrics']:
        with op.batch_alter_table(foreign, naming_convention=conv) as batch_op:
            batch_op.add_column(sa.Column('datasource_name', sa.String(255)))
            batch_op.create_foreign_key(f'fk_{foreign}_datasource_name_datasources', 'datasources', ['datasource_name'], ['datasource_name'])
        table = sa.Table(foreign, sa.MetaData(), sa.Column('id', sa.Integer, primary_key=True), sa.Column('datasource_name', sa.String(255)), sa.Column('datasource_id', sa.Integer))
        for datasource in bind.execute(datasources.select()):
            bind.execute(table.update().where(table.c.datasource_id == datasource.id).values(datasource_name=datasource.datasource_name))
        with op.batch_alter_table(foreign, naming_convention=conv) as batch_op:
            batch_op.drop_constraint(f'fk_{foreign}_datasource_id_datasources', type_='foreignkey')
            batch_op.drop_column('datasource_id')
    with op.batch_alter_table('datasources', naming_convention=conv) as batch_op:
        batch_op.drop_constraint(generic_find_fk_constraint_name('datasources', {'cluster_name'}, 'clusters', insp) or 'fk_datasources_cluster_name_clusters', type_='foreignkey')
        batch_op.drop_constraint(generic_find_uq_constraint_name('datasources', {'cluster_name', 'datasource_name'}, insp) or 'uq_datasources_cluster_name', type_='unique')
        batch_op.create_foreign_key(f'fk_{foreign}_datasource_id_datasources', 'clusters', ['cluster_name'], ['cluster_name'])