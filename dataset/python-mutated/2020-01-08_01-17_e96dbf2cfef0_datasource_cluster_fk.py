"""datasource_cluster_fk

Revision ID: e96dbf2cfef0
Revises: 817e1c9b09d0
Create Date: 2020-01-08 01:17:40.127610

"""
import sqlalchemy as sa
from alembic import op
from superset import db
from superset.utils.core import generic_find_fk_constraint_name, generic_find_uq_constraint_name
revision = 'e96dbf2cfef0'
down_revision = '817e1c9b09d0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer()))
    metadata = sa.MetaData(bind=bind)
    datasources = sa.Table('datasources', metadata, autoload=True)
    clusters = sa.Table('clusters', metadata, autoload=True)
    statement = datasources.update().values(cluster_id=sa.select([clusters.c.id]).where(datasources.c.cluster_name == clusters.c.cluster_name).as_scalar())
    bind.execute(statement)
    with op.batch_alter_table('datasources') as batch_op:
        fk_constraint_name = generic_find_fk_constraint_name('datasources', {'cluster_name'}, 'clusters', insp)
        uq_constraint_name = generic_find_uq_constraint_name('datasources', {'cluster_name', 'datasource_name'}, insp)
        batch_op.drop_constraint(fk_constraint_name, type_='foreignkey')
        batch_op.drop_constraint(uq_constraint_name, type_='unique')
        batch_op.drop_column('cluster_name')
        batch_op.alter_column('cluster_id', existing_type=sa.Integer, nullable=False)
        batch_op.create_unique_constraint('uq_datasources_cluster_id', ['cluster_id', 'datasource_name'])
        batch_op.create_foreign_key('fk_datasources_cluster_id_clusters', 'clusters', ['cluster_id'], ['id'])

def downgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.add_column(sa.Column('cluster_name', sa.String(250)))
    metadata = sa.MetaData(bind=bind)
    datasources = sa.Table('datasources', metadata, autoload=True)
    clusters = sa.Table('clusters', metadata, autoload=True)
    statement = datasources.update().values(cluster_name=sa.select([clusters.c.cluster_name]).where(datasources.c.cluster_id == clusters.c.id).as_scalar())
    bind.execute(statement)
    with op.batch_alter_table('datasources') as batch_op:
        fk_constraint_name = generic_find_fk_constraint_name('datasources', {'id'}, 'clusters', insp)
        uq_constraint_name = generic_find_uq_constraint_name('datasources', {'cluster_id', 'datasource_name'}, insp)
        batch_op.drop_constraint(fk_constraint_name, type_='foreignkey')
        batch_op.drop_constraint(uq_constraint_name, type_='unique')
        batch_op.drop_column('cluster_id')
        batch_op.alter_column('cluster_name', existing_type=sa.String(250), nullable=False)
        batch_op.create_unique_constraint('uq_datasources_cluster_name', ['cluster_name', 'datasource_name'])
        batch_op.create_foreign_key('fk_datasources_cluster_name_clusters', 'clusters', ['cluster_name'], ['cluster_name'])