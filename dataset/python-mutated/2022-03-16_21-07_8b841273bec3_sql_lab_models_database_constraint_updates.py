"""sql_lab_models_database_constraint_updates

Revision ID: 8b841273bec3
Revises: 2ed890b36b94
Create Date: 2022-03-16 21:07:48.768425

"""
revision = '8b841273bec3'
down_revision = '2ed890b36b94'
import sqlalchemy as sa
from alembic import op
from superset.utils.core import generic_find_fk_constraint_name

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('tab_state') as batch_op:
        table_schema_id_constraint = generic_find_fk_constraint_name('tab_state', {'id'}, 'dbs', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        table_schema_id_constraint = generic_find_fk_constraint_name('tab_state', {'client_id'}, 'query', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        batch_op.create_foreign_key('tab_state_database_id_fkey', 'dbs', ['database_id'], ['id'], ondelete='CASCADE')
        batch_op.create_foreign_key('tab_state_latest_query_id_fkey', 'query', ['latest_query_id'], ['client_id'], ondelete='SET NULL')
    with op.batch_alter_table('table_schema') as batch_op:
        table_schema_id_constraint = generic_find_fk_constraint_name('table_schema', {'id'}, 'dbs', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        batch_op.create_foreign_key('table_schema_database_id_fkey', 'dbs', ['database_id'], ['id'], ondelete='CASCADE')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    insp = sa.engine.reflection.Inspector.from_engine(bind)
    with op.batch_alter_table('tab_state') as batch_op:
        table_schema_id_constraint = generic_find_fk_constraint_name('tab_state', {'id'}, 'dbs', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        table_schema_id_constraint = generic_find_fk_constraint_name('tab_state', {'client_id'}, 'query', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        batch_op.create_foreign_key('tab_state_database_id_fkey', 'dbs', ['database_id'], ['id'])
        batch_op.create_foreign_key('tab_state_latest_query_id_fkey', 'query', ['latest_query_id'], ['client_id'])
    with op.batch_alter_table('table_schema') as batch_op:
        table_schema_id_constraint = generic_find_fk_constraint_name('table_schema', {'id'}, 'dbs', insp)
        if table_schema_id_constraint:
            batch_op.drop_constraint(table_schema_id_constraint, type_='foreignkey')
        batch_op.create_foreign_key('table_schema_database_id_fkey', 'dbs', ['database_id'], ['id'])