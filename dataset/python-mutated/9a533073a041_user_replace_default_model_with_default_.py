"""user replace default_model with default_preset

Revision ID: 9a533073a041
Revises:
Create Date: 2023-05-11 19:19:27.659031

"""
from alembic import op
revision = '9a533073a041'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('user', 'default_model', new_column_name='default_preset')
    op.execute("UPDATE user SET default_preset = '';")