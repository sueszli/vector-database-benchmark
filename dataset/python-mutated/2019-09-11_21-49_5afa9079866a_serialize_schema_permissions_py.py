"""serialize_schema_permissions.py

Revision ID: 5afa9079866a
Revises: db4b49eb0782
Create Date: 2019-09-11 21:49:00.608346

"""
from alembic import op
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from superset import db
revision = '5afa9079866a'
down_revision = 'db4b49eb0782'
Base = declarative_base()

class Sqlatable(Base):
    __tablename__ = 'tables'
    id = Column(Integer, primary_key=True)
    perm = Column(String(1000))
    schema_perm = Column(String(1000))
    schema = Column(String(255))
    database_id = Column(Integer, ForeignKey('dbs.id'), nullable=False)
    database = relationship('Database', foreign_keys=[database_id])

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    datasource_id = Column(Integer)
    datasource_type = Column(String(200))
    schema_perm = Column(String(1000))

class Database(Base):
    __tablename__ = 'dbs'
    id = Column(Integer, primary_key=True)
    database_name = Column(String(250))
    verbose_name = Column(String(250), unique=True)

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('datasources', Column('schema_perm', String(length=1000), nullable=True))
    op.add_column('slices', Column('schema_perm', String(length=1000), nullable=True))
    op.add_column('tables', Column('schema_perm', String(length=1000), nullable=True))
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for t in session.query(Sqlatable).all():
        db_name = t.database.verbose_name if t.database.verbose_name else t.database.database_name
        if t.schema:
            t.schema_perm = f'[{db_name}].[{t.schema}]'
            table_slices = session.query(Slice).filter_by(datasource_type='table').filter_by(datasource_id=t.id).all()
            for s in table_slices:
                s.schema_perm = t.schema_perm
    session.commit()

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('tables', 'schema_perm')
    op.drop_column('datasources', 'schema_perm')
    op.drop_column('slices', 'schema_perm')