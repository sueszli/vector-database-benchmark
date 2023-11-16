"""
handle_null_on_projects_total_size

Revision ID: d83f20495c10
Revises: 48def930fcfd
Create Date: 2019-08-10 20:47:10.155339
"""
from alembic import op
revision = 'd83f20495c10'
down_revision = '48def930fcfd'

def upgrade():
    if False:
        return 10
    op.execute("CREATE OR REPLACE FUNCTION projects_total_size()\n        RETURNS TRIGGER AS $$\n        DECLARE\n            _release_id uuid;\n            _project_id uuid;\n\n        BEGIN\n            IF TG_OP = 'INSERT' THEN\n                _release_id := NEW.release_id;\n            ELSEIF TG_OP = 'UPDATE' THEN\n                _release_id := NEW.release_id;\n            ELSIF TG_OP = 'DELETE' THEN\n                _release_id := OLD.release_id;\n            END IF;\n            _project_id := (SELECT project_id\n                            FROM releases\n                            WHERE releases.id=_release_id);\n            UPDATE projects\n            SET total_size=t.project_total_size\n            FROM (\n            SELECT COALESCE(SUM(release_files.size), 0) AS project_total_size\n            FROM release_files WHERE release_id IN\n                (SELECT id FROM releases WHERE releases.project_id = _project_id)\n            ) AS t\n            WHERE id=_project_id;\n            RETURN NULL;\n        END;\n        $$ LANGUAGE plpgsql;\n        ")

def downgrade():
    if False:
        i = 10
        return i + 15
    op.execute("CREATE OR REPLACE FUNCTION projects_total_size()\n        RETURNS TRIGGER AS $$\n        DECLARE\n            _release_id uuid;\n            _project_id uuid;\n\n        BEGIN\n            IF TG_OP = 'INSERT' THEN\n                _release_id := NEW.release_id;\n            ELSEIF TG_OP = 'UPDATE' THEN\n                _release_id := NEW.release_id;\n            ELSIF TG_OP = 'DELETE' THEN\n                _release_id := OLD.release_id;\n            END IF;\n            _project_id := (SELECT project_id\n                            FROM releases\n                            WHERE releases.id=_release_id);\n            UPDATE projects\n            SET total_size=t.project_total_size\n            FROM (\n            SELECT SUM(release_files.size) AS project_total_size\n            FROM release_files WHERE release_id IN\n                (SELECT id FROM releases WHERE releases.project_id = _project_id)\n            ) AS t\n            WHERE id=_project_id;\n            RETURN NULL;\n        END;\n        $$ LANGUAGE plpgsql;\n        ")