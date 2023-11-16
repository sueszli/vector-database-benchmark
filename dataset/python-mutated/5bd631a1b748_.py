"""Updating fixed flag in the issue table to be not nullable.

Revision ID: 5bd631a1b748
Revises: 4ac52090a637
Create Date: 2017-09-26 11:05:23.060909

"""
revision = '5bd631a1b748'
down_revision = '4ac52090a637'
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Session = sessionmaker()
Base = declarative_base()

class ItemAudit(Base):
    __tablename__ = 'itemaudit'
    id = sa.Column(sa.Integer, primary_key=True)
    fixed = sa.Column(sa.Boolean)

def upgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = Session(bind=bind)
    session.query(ItemAudit).filter(ItemAudit.fixed == None).update(dict(fixed=False))
    session.commit()
    op.alter_column('itemaudit', 'fixed', nullable=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('itemaudit', 'fixed', nullable=True)