"""Materializing permission

Revision ID: c3a8f8611885
Revises: 4fa88fe24e94
Create Date: 2016-04-25 08:54:04.303859

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from superset import db
revision = 'c3a8f8611885'
down_revision = '4fa88fe24e94'
Base = declarative_base()

class Slice(Base):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    slice_name = Column(String(250))
    druid_datasource_id = Column(Integer, ForeignKey('datasources.id'))
    table_id = Column(Integer, ForeignKey('tables.id'))
    perm = Column(String(2000))

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    op.add_column('slices', sa.Column('perm', sa.String(length=2000), nullable=True))
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        if slc.datasource:
            slc.perm = slc.datasource.perm
            session.merge(slc)
            session.commit()
    db.session.close()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('slices') as batch_op:
        batch_op.drop_column('perm')