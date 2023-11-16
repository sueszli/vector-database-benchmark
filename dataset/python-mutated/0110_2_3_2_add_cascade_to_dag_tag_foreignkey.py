"""Add cascade to dag_tag foreign key

Revision ID: 3c94c427fdf6
Revises: 1de7bc13c950
Create Date: 2022-05-03 09:47:41.957710

"""
from __future__ import annotations
from alembic import op
from sqlalchemy import inspect
from airflow.migrations.utils import get_mssql_table_constraints
revision = '3c94c427fdf6'
down_revision = '1de7bc13c950'
branch_labels = None
depends_on = None
airflow_version = '2.3.2'

def upgrade():
    if False:
        return 10
    'Apply Add cascade to dag_tag foreignkey'
    conn = op.get_bind()
    if conn.dialect.name in ['sqlite', 'mysql']:
        inspector = inspect(conn.engine)
        foreignkey = inspector.get_foreign_keys('dag_tag')
        with op.batch_alter_table('dag_tag') as batch_op:
            batch_op.drop_constraint(foreignkey[0]['name'], type_='foreignkey')
            batch_op.create_foreign_key('dag_tag_dag_id_fkey', 'dag', ['dag_id'], ['dag_id'], ondelete='CASCADE')
    else:
        with op.batch_alter_table('dag_tag') as batch_op:
            if conn.dialect.name == 'mssql':
                constraints = get_mssql_table_constraints(conn, 'dag_tag')
                (Fk, _) = constraints['FOREIGN KEY'].popitem()
                batch_op.drop_constraint(Fk, type_='foreignkey')
            if conn.dialect.name == 'postgresql':
                batch_op.drop_constraint('dag_tag_dag_id_fkey', type_='foreignkey')
            batch_op.create_foreign_key('dag_tag_dag_id_fkey', 'dag', ['dag_id'], ['dag_id'], ondelete='CASCADE')

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Add cascade to dag_tag foreignkey'
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite':
        with op.batch_alter_table('dag_tag') as batch_op:
            batch_op.drop_constraint('dag_tag_dag_id_fkey', type_='foreignkey')
            batch_op.create_foreign_key('fk_dag_tag_dag_id_dag', 'dag', ['dag_id'], ['dag_id'])
    else:
        with op.batch_alter_table('dag_tag') as batch_op:
            batch_op.drop_constraint('dag_tag_dag_id_fkey', type_='foreignkey')
            batch_op.create_foreign_key(None, 'dag', ['dag_id'], ['dag_id'])