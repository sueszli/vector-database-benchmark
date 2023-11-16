"""sqllab_setting_defaults

Revision ID: eca4694defa7
Revises: 5e4a03ef0bf0
Create Date: 2016-09-22 11:31:50.543820

"""
from alembic import op
from sqlalchemy import Boolean, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from superset import db
revision = 'eca4694defa7'
down_revision = '5e4a03ef0bf0'
Base = declarative_base()

class Database(Base):
    """An ORM object that stores Database related information"""
    __tablename__ = 'dbs'
    id = Column(Integer, primary_key=True)
    allow_run_sync = Column(Boolean, default=True)

def upgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for obj in session.query(Database).all():
        obj.allow_run_sync = True
    session.commit()
    session.close()

def downgrade():
    if False:
        print('Hello World!')
    pass