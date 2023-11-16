"""Add missing auto-increment to columns on FAB tables

Revision ID: b0d31815b5a6
Revises: ecb43d2a1842
Create Date: 2022-10-05 13:16:45.638490

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'b0d31815b5a6'
down_revision = 'ecb43d2a1842'
branch_labels = None
depends_on = None
airflow_version = '2.4.2'

def upgrade():
    if False:
        i = 10
        return i + 15
    "Apply migration.\n\n    If these columns are already of the right type (i.e. created by our\n    migration in 1.10.13 rather than FAB itself in an earlier version), this\n    migration will issue an alter statement to change them to what they already\n    are -- i.e. its a no-op.\n\n    These tables are small (100 to low 1k rows at most), so it's not too costly\n    to change them.\n    "
    conn = op.get_bind()
    if conn.dialect.name in ['mssql', 'sqlite']:
        return
    for table in ('ab_permission', 'ab_view_menu', 'ab_role', 'ab_permission_view', 'ab_permission_view_role', 'ab_user', 'ab_user_role', 'ab_register_user'):
        with op.batch_alter_table(table) as batch:
            kwargs = {}
            if conn.dialect.name == 'postgresql':
                kwargs['server_default'] = sa.Sequence(f'{table}_id_seq').next_value()
            else:
                kwargs['autoincrement'] = True
            batch.alter_column('id', existing_type=sa.Integer(), existing_nullable=False, **kwargs)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply add_missing_autoinc_fab'