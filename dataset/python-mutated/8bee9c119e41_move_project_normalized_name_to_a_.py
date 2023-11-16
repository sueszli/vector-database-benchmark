"""
Move Project.normalized_name to a regular column

Revision ID: 8bee9c119e41
Revises: 5a095c98f812
Create Date: 2022-06-27 14:48:16.619143
"""
import sqlalchemy as sa
from alembic import op
revision = '8bee9c119e41'
down_revision = '5a095c98f812'

def upgrade():
    if False:
        return 10
    op.add_column('projects', sa.Column('normalized_name', sa.Text(), nullable=True))
    op.execute('\n        UPDATE projects\n        SET normalized_name = normalize_pep426_name(name)\n        ')
    op.alter_column('projects', 'normalized_name', nullable=False)
    op.create_unique_constraint(None, 'projects', ['normalized_name'])
    op.execute('DROP INDEX project_name_pep426_normalized')
    op.execute(' CREATE OR REPLACE FUNCTION maintain_projects_normalized_name()\n            RETURNS TRIGGER AS $$\n                BEGIN\n                    NEW.normalized_name :=  normalize_pep426_name(NEW.name);\n                    RETURN NEW;\n                END;\n            $$\n            LANGUAGE plpgsql\n        ')
    op.execute(' CREATE TRIGGER projects_update_normalized_name\n            BEFORE INSERT OR UPDATE OF name ON projects\n            FOR EACH ROW\n            EXECUTE PROCEDURE maintain_projects_normalized_name()\n        ')

def downgrade():
    if False:
        return 10
    op.execute(' CREATE UNIQUE INDEX project_name_pep426_normalized\n            ON projects\n            (normalize_pep426_name(name))\n        ')
    op.drop_constraint(None, 'projects', type_='unique')
    op.drop_column('projects', 'normalized_name')