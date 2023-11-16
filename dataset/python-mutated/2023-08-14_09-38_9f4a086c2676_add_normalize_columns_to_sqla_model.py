"""add_normalize_columns_to_sqla_model

Revision ID: 9f4a086c2676
Revises: 4448fa6deeb1
Create Date: 2023-08-14 09:38:11.897437

"""
revision = '9f4a086c2676'
down_revision = '4448fa6deeb1'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from superset import db
from superset.migrations.shared.utils import paginated_update
Base = declarative_base()

class SqlaTable(Base):
    __tablename__ = 'tables'
    id = sa.Column(sa.Integer, primary_key=True)
    normalize_columns = sa.Column(sa.Boolean())

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('tables', sa.Column('normalize_columns', sa.Boolean(), nullable=True, default=False, server_default=sa.false()))
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for table in paginated_update(session.query(SqlaTable)):
        table.normalize_columns = True

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('tables', 'normalize_columns')