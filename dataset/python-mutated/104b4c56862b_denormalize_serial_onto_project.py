"""
Denormalize serial onto project

Revision ID: 104b4c56862b
Revises: fb3278418206
Create Date: 2016-05-04 21:47:04.133779
"""
import sqlalchemy as sa
from alembic import op
revision = '104b4c56862b'
down_revision = 'fb3278418206'

def upgrade():
    if False:
        while True:
            i = 10
    op.execute('LOCK TABLE packages IN EXCLUSIVE MODE')
    op.execute('LOCK TABLE journals IN EXCLUSIVE MODE')
    op.add_column('packages', sa.Column('last_serial', sa.Integer(), nullable=True, server_default=sa.text('0')))
    op.execute(' UPDATE packages\n            SET last_serial = j.last_serial\n            FROM (\n                SELECT name,\n                       max(id) as last_serial\n                FROM journals\n                GROUP BY name\n            ) as j\n            WHERE j.name = packages.name\n        ')
    op.alter_column('packages', 'last_serial', nullable=False)
    op.execute(" CREATE OR REPLACE FUNCTION maintain_project_last_serial()\n            RETURNS TRIGGER AS $$\n            DECLARE\n                targeted_name text;\n            BEGIN\n                IF TG_OP = 'INSERT' THEN\n                    targeted_name := NEW.name;\n                ELSEIF TG_OP = 'UPDATE' THEN\n                    targeted_name := NEW.name;\n                ELSIF TG_OP = 'DELETE' THEN\n                    targeted_name := OLD.name;\n                END IF;\n\n                UPDATE packages\n                SET last_serial = j.last_serial\n                FROM (\n                    SELECT max(id) as last_serial\n                    FROM journals\n                    WHERE journals.name = targeted_name\n                ) as j\n                WHERE packages.name = targeted_name;\n\n                RETURN NULL;\n            END;\n            $$ LANGUAGE plpgsql;\n        ")
    op.execute(' CREATE TRIGGER update_project_last_serial\n            AFTER INSERT OR UPDATE OR DELETE ON journals\n            FOR EACH ROW EXECUTE PROCEDURE maintain_project_last_serial();\n        ')

def downgrade():
    if False:
        i = 10
        return i + 15
    raise RuntimeError('Order No. 227 - Ни шагу назад!')