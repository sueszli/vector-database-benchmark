"""change_table_unique_constraint

Revision ID: b4456560d4f3
Revises: bb51420eaf83
Create Date: 2016-04-15 08:31:26.249591

"""
from alembic import op
revision = 'b4456560d4f3'
down_revision = 'bb51420eaf83'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    try:
        op.drop_constraint('tables_table_name_key', 'tables', type_='unique')
        op.create_unique_constraint('_customer_location_uc', 'tables', ['database_id', 'schema', 'table_name'])
    except Exception:
        pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    try:
        op.drop_constraint('_customer_location_uc', 'tables', type_='unique')
    except Exception:
        pass