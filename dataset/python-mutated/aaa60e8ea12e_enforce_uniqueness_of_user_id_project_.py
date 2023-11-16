"""
enforce uniqueness of user_id, project_id on roles

Revision ID: aaa60e8ea12e
Revises: 5c029d9ef925
Create Date: 2020-03-04 21:56:32.651065
"""
from alembic import op
revision = 'aaa60e8ea12e'
down_revision = '5c029d9ef925'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('\n        DELETE FROM roles\n        WHERE id IN (\n            SELECT id FROM (\n                SELECT id,\n                ROW_NUMBER() OVER (\n                    PARTITION BY project_id, user_id ORDER BY role_name DESC\n                ) as row_num\n                FROM roles\n            ) t\n            WHERE t.row_num > 1\n        )\n        RETURNING *\n        ')
    op.create_unique_constraint('_roles_user_project_uc', 'roles', ['user_id', 'project_id'])

def downgrade():
    if False:
        print('Hello World!')
    op.drop_constraint('_roles_user_project_uc', 'roles', type_='unique')