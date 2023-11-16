"""rm_time_range_endpoints_from_qc_3

Revision ID: ad07e4fdbaba
Revises: cecc6bf46990
Create Date: 2022-04-18 11:20:47.390901

"""
revision = 'ad07e4fdbaba'
down_revision = 'cecc6bf46990'
import json
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = sa.Column(sa.Integer, primary_key=True)
    query_context = sa.Column(sa.Text)
    slice_name = sa.Column(sa.String(250))

def upgrade_slice(slc: Slice):
    if False:
        while True:
            i = 10
    try:
        query_context = json.loads(slc.query_context)
    except json.decoder.JSONDecodeError:
        return
    query_context.get('form_data', {}).pop('time_range_endpoints', None)
    if query_context.get('queries'):
        queries = query_context['queries']
        for query in queries:
            query.get('extras', {}).pop('time_range_endpoints', None)
    slc.query_context = json.dumps(query_context)
    return slc

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices_updated = 0
    for slc in session.query(Slice).filter(Slice.query_context.like('%time_range_endpoints%')).all():
        updated_slice = upgrade_slice(slc)
        if updated_slice:
            slices_updated += 1
    print(f'slices updated with no time_range_endpoints: {slices_updated}')
    session.commit()
    session.close()

def downgrade():
    if False:
        print('Hello World!')
    pass