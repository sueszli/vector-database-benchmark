"""add run status index.

Revision ID: 3e71cf573ba6
Revises: 6d366a41b4be
Create Date: 2021-02-23 16:00:45.689578

"""
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "3e71cf573ba6"
down_revision = "6d366a41b4be"
branch_labels = None
depends_on = None


def upgrade():
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if "runs" in has_tables:
        indices = [x.get("name") for x in inspector.get_indexes("runs")]
        if "idx_run_status" not in indices:
            op.create_index("idx_run_status", "runs", ["status"], unique=False)


def downgrade():
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if "runs" in has_tables:
        indices = [x.get("name") for x in inspector.get_indexes("runs")]
        if "idx_run_status" in indices:
            op.drop_index("idx_run_status", "runs")
