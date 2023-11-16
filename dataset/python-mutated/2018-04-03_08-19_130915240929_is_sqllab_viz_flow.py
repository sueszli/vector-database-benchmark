"""is_sqllab_view

Revision ID: 130915240929
Revises: f231d82b9b26
Create Date: 2018-04-03 08:19:34.098789

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from superset import db
revision = '130915240929'
down_revision = 'f231d82b9b26'
Base = declarative_base()

class Table(Base):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'tables'
    id = sa.Column(sa.Integer, primary_key=True)
    sql = sa.Column(sa.Text)
    is_sqllab_view = sa.Column(sa.Boolean())

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    op.add_column('tables', sa.Column('is_sqllab_view', sa.Boolean(), nullable=True, default=False, server_default=sa.false()))
    session = db.Session(bind=bind)
    for tbl in session.query(Table).all():
        if tbl.sql:
            tbl.is_sqllab_view = True
    session.commit()
    db.session.close()

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('tables', 'is_sqllab_view')