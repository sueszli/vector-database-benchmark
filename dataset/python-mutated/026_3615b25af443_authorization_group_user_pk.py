"""026 Authorization group user pk

Revision ID: 3615b25af443
Revises: b581622ad327
Create Date: 2018-09-04 18:48:57.988110

"""
from alembic import op
from ckan.migration import skip_based_on_legacy_engine_version
revision = '3615b25af443'
down_revision = 'b581622ad327'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    if skip_based_on_legacy_engine_version(op, __name__):
        return
    pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass