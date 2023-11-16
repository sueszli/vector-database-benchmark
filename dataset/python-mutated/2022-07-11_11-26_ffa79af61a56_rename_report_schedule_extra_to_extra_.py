"""rename report_schedule.extra to extra_json

So we can reuse the ExtraJSONMixin

Revision ID: ffa79af61a56
Revises: 409c7b420ab0
Create Date: 2022-07-11 11:26:00.010714

"""
revision = 'ffa79af61a56'
down_revision = '409c7b420ab0'
from alembic import op
from sqlalchemy.types import Text

def upgrade():
    if False:
        i = 10
        return i + 15
    op.alter_column('report_schedule', 'extra', new_column_name='extra_json', existing_type=Text, existing_nullable=True)

def downgrade():
    if False:
        print('Hello World!')
    op.alter_column('report_schedule', 'extra_json', new_column_name='extra', existing_type=Text, existing_nullable=True)