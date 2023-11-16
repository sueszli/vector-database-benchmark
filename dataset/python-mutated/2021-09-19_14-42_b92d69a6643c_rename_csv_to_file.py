"""rename_csv_to_file

Revision ID: b92d69a6643c
Revises: aea15018d53b
Create Date: 2021-09-19 14:42:20.130368

"""
revision = 'b92d69a6643c'
down_revision = 'aea15018d53b'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

def upgrade():
    if False:
        i = 10
        return i + 15
    try:
        with op.batch_alter_table('dbs') as batch_op:
            batch_op.alter_column('allow_csv_upload', new_column_name='allow_file_upload', existing_type=sa.Boolean())
    except (sa.exc.OperationalError, sa.exc.DatabaseError):
        bind = op.get_bind()
        inspector = Inspector.from_engine(bind)
        check_constraints = inspector.get_check_constraints('dbs')
        for check_constraint in check_constraints:
            if 'allow_csv_upload' in check_constraint['sqltext']:
                name = check_constraint['name']
                op.drop_constraint(name, table_name='dbs', type_='check')
        with op.batch_alter_table('dbs') as batch_op:
            batch_op.alter_column('allow_csv_upload', new_column_name='allow_file_upload', existing_type=sa.Boolean())

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.alter_column('allow_file_upload', new_column_name='allow_csv_upload', existing_type=sa.Boolean())