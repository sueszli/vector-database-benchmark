"""0.10.0 create new schedule tables.

Revision ID: 493871843165
Revises: 942138e33bf9
Create Date: 2021-01-13 14:43:03.678784

"""
from dagster._core.storage.migration.utils import create_0_10_0_schedule_tables
revision = '493871843165'
down_revision = '942138e33bf9'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    create_0_10_0_schedule_tables()

def downgrade():
    if False:
        print('Hello World!')
    pass