"""add granularity to charts where missing

Revision ID: 070c043f2fdb
Revises: 41ce8799acc3
Create Date: 2021-02-04 09:34:13.608891

"""
revision = '070c043f2fdb'
down_revision = '41ce8799acc3'
import json
from alembic import op
from sqlalchemy import and_, Boolean, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)
    datasource_id = Column(Integer)
    datasource_type = Column(String(200))

class SqlaTable(Base):
    __tablename__ = 'tables'
    id = Column(Integer, primary_key=True)
    main_dttm_col = Column(String(250))

class TableColumn(Base):
    __tablename__ = 'table_columns'
    id = Column(Integer, primary_key=True)
    table_id = Column(Integer)
    is_dttm = Column(Boolean)
    column_name = Column(String(255))

def upgrade():
    if False:
        while True:
            i = 10
    "\n    Adds the granularity param to charts without it populated. This is required for\n    time range filtering to work properly. Uses the following approach:\n\n    - Find all charts without a granularity or granularity_sqla param.\n    - Get the dataset that backs the chart.\n    - If the dataset has the main dttm column set, use it.\n    - Otherwise, find all the dttm columns in the dataset and use the first one (this\n      matches the behavior of Explore view on the frontend)\n    - If no dttm columns exist in the dataset, don't change the chart.\n    "
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices_changed = 0
    for slc in session.query(Slice).filter(and_(Slice.datasource_type == 'table', Slice.params.notlike('%"granularity%'))).all():
        try:
            params = json.loads(slc.params)
            if 'granularity' in params or 'granularity_sqla' in params:
                continue
            table = session.query(SqlaTable).get(slc.datasource_id)
            if not table:
                continue
            if table.main_dttm_col:
                params['granularity'] = table.main_dttm_col
                slc.params = json.dumps(params, sort_keys=True)
                print(f'Set granularity for slice {slc.id} to {table.main_dttm_col}')
                slices_changed += 1
                continue
            table_columns = session.query(TableColumn).filter(TableColumn.table_id == table.id).filter(TableColumn.is_dttm == True).all()
            if len(table_columns):
                params['granularity'] = table_columns[0].column_name
                slc.params = json.dumps(params, sort_keys=True)
                print(f'Set granularity for slice {slc.id} to {table_columns[0].column_name}')
                slices_changed += 1
        except Exception as e:
            print(e)
            print(f'Parsing params for slice {slc.id} failed.')
            pass
    print(f'{slices_changed} slices altered')
    session.commit()
    session.close()

def downgrade():
    if False:
        while True:
            i = 10
    "\n    It's impossible to downgrade this migration.\n    "
    pass