"""add_created_by

Revision ID: 6d548701edef
Revises: ad4b1b4d1e9d
Create Date: 2022-10-19 09:39:02.371032

"""
import sqlalchemy as sa
from alembic import op

import prefect

# revision identifiers, used by Alembic.
revision = "6d548701edef"
down_revision = "ad4b1b4d1e9d"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "deployment",
        sa.Column(
            "created_by",
            prefect.server.utilities.database.Pydantic(
                prefect.server.schemas.core.CreatedBy
            ),
            nullable=True,
        ),
    )
    op.add_column(
        "deployment",
        sa.Column(
            "updated_by",
            prefect.server.utilities.database.Pydantic(
                prefect.server.schemas.core.UpdatedBy
            ),
            nullable=True,
        ),
    )
    op.add_column(
        "flow_run",
        sa.Column(
            "created_by",
            prefect.server.utilities.database.Pydantic(
                prefect.server.schemas.core.CreatedBy
            ),
            nullable=True,
        ),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("flow_run", "created_by")
    op.drop_column("deployment", "updated_by")
    op.drop_column("deployment", "created_by")
    # ### end Alembic commands ###
