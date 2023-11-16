"""Fix schema perm for datasets

Revision ID: 0769ef90fddd
Revises: ee179a490af9
Create Date: 2023-08-02 15:23:58.242396

"""
revision = '0769ef90fddd'
down_revision = 'ee179a490af9'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class SqlaTable(Base):
    __tablename__ = 'tables'
    id = sa.Column(sa.Integer, primary_key=True)
    schema = sa.Column(sa.String(255))
    schema_perm = sa.Column(sa.String(1000))
    database_id = sa.Column(sa.Integer, sa.ForeignKey('dbs.id'))

class Slice(Base):
    __tablename__ = 'slices'
    id = sa.Column(sa.Integer, primary_key=True)
    schema_perm = sa.Column(sa.String(1000))
    datasource_id = sa.Column(sa.Integer)

class Database(Base):
    __tablename__ = 'dbs'
    id = sa.Column(sa.Integer, primary_key=True)
    database_name = sa.Column(sa.String(250))

def fix_datasets_schema_perm(session):
    if False:
        print('Hello World!')
    for result in session.query(SqlaTable, Database.database_name).join(Database).filter(SqlaTable.schema.isnot(None)).filter(SqlaTable.schema_perm != sa.func.concat('[', Database.database_name, '].[', SqlaTable.schema, ']')):
        result.SqlaTable.schema_perm = f'[{result.database_name}].[{result.SqlaTable.schema}]'

def fix_charts_schema_perm(session):
    if False:
        while True:
            i = 10
    for result in session.query(Slice, SqlaTable, Database.database_name).join(SqlaTable, Slice.datasource_id == SqlaTable.id).join(Database, SqlaTable.database_id == Database.id).filter(SqlaTable.schema.isnot(None)).filter(Slice.schema_perm != sa.func.concat('[', Database.database_name, '].[', SqlaTable.schema, ']')):
        result.Slice.schema_perm = f'[{result.database_name}].[{result.SqlaTable.schema}]'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    if isinstance(bind.dialect, SQLiteDialect):
        return
    fix_datasets_schema_perm(session)
    fix_charts_schema_perm(session)
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    pass