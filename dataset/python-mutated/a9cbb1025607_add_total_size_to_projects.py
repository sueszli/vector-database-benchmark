"""
add_total_size_to_projects

Revision ID: a9cbb1025607
Revises: cdb2915fda5c
Create Date: 2019-06-15 09:01:46.641354
"""
import sqlalchemy as sa
from alembic import op
revision = 'a9cbb1025607'
down_revision = 'cdb2915fda5c'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('projects', sa.Column('total_size', sa.BigInteger(), server_default=sa.text('0')))
    op.execute("CREATE OR REPLACE FUNCTION projects_total_size()\n        RETURNS TRIGGER AS $$\n        DECLARE\n            _release_id uuid;\n            _project_id uuid;\n\n        BEGIN\n            IF TG_OP = 'INSERT' THEN\n                _release_id := NEW.release_id;\n            ELSEIF TG_OP = 'UPDATE' THEN\n                _release_id := NEW.release_id;\n            ELSIF TG_OP = 'DELETE' THEN\n                _release_id := OLD.release_id;\n            END IF;\n            _project_id := (SELECT project_id\n                            FROM releases\n                            WHERE releases.id=_release_id);\n            UPDATE projects\n            SET total_size=t.project_total_size\n            FROM (\n            SELECT SUM(release_files.size) AS project_total_size\n            FROM release_files WHERE release_id IN\n                (SELECT id FROM releases WHERE releases.project_id = _project_id)\n            ) AS t\n            WHERE id=_project_id;\n            RETURN NULL;\n        END;\n        $$ LANGUAGE plpgsql;\n        ")
    op.execute('CREATE TRIGGER update_project_total_size\n            AFTER INSERT OR UPDATE OR DELETE ON release_files\n            FOR EACH ROW EXECUTE PROCEDURE projects_total_size();\n        ')
    op.execute('WITH project_totals AS (\n                SELECT\n                    p.id as project_id,\n                    sum(size) as project_total\n                FROM\n                    release_files rf\n                    JOIN releases r on rf.release_id = r.id\n                    JOIN projects p on r.project_id = p.id\n                GROUP BY\n                    p.id\n            )\n            UPDATE projects AS p\n            SET total_size = project_totals.project_total\n            FROM project_totals\n            WHERE project_totals.project_id = p.id;\n        ')

def downgrade():
    if False:
        while True:
            i = 10
    op.execute('DROP TRIGGER update_project_total_size ON release_files;')
    op.execute('DROP FUNCTION projects_total_size;')
    op.drop_column('projects', 'total_size')