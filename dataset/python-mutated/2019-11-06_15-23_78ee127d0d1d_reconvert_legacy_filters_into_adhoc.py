"""reconvert legacy filters into adhoc

Revision ID: 78ee127d0d1d
Revises: c2acd2cf3df2
Create Date: 2019-11-06 15:23:26.497876

"""
revision = '78ee127d0d1d'
down_revision = 'c2acd2cf3df2'
import copy
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
from superset.utils.core import convert_legacy_filters_into_adhoc, split_adhoc_filters_into_base_filters
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        if slc.params:
            try:
                source = json.loads(slc.params)
                target = copy.deepcopy(source)
                convert_legacy_filters_into_adhoc(target)
                if source != target:
                    slc.params = json.dumps(target, sort_keys=True)
            except Exception as ex:
                logging.warn(ex)
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    pass