"""empty message

Revision ID: f231d82b9b26
Revises: e68c4473c581
Create Date: 2018-03-20 19:47:54.991259

"""
import sqlalchemy as sa
from alembic import op
from superset.utils.core import generic_find_uq_constraint_name
revision = 'f231d82b9b26'
down_revision = 'e68c4473c581'
conv = {'uq': 'uq_%(table_name)s_%(column_0_name)s'}
names = {'columns': 'column_name', 'metrics': 'metric_name'}

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('metrics', naming_convention=conv) as batch_op:
        batch_op.alter_column('metric_name', existing_type=sa.String(length=512), type_=sa.String(length=255), existing_nullable=True)
    for (table, column) in names.items():
        with op.batch_alter_table(table, naming_convention=conv) as batch_op:
            batch_op.create_unique_constraint(f'uq_{table}_{column}', [column, 'datasource_id'])

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('metrics', naming_convention=conv) as batch_op:
        batch_op.alter_column('metric_name', existing_type=sa.String(length=255), type_=sa.String(length=512), existing_nullable=True)
    for (table, column) in names.items():
        with op.batch_alter_table(table, naming_convention=conv) as batch_op:
            batch_op.drop_constraint(generic_find_uq_constraint_name(table, {column, 'datasource_id'}, insp) or f'uq_{table}_{column}', type_='unique')