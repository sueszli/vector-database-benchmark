"""chart-ds-constraint

Revision ID: 7e67aecbf3f1
Revises: b5ea9d343307
Create Date: 2023-03-27 12:30:01.164594

"""
revision = '7e67aecbf3f1'
down_revision = '07f9a902af1b'
import json
import logging
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()
logger = logging.getLogger(__name__)

class Slice(Base):
    __tablename__ = 'slices'
    id = sa.Column(sa.Integer, primary_key=True)
    params = sa.Column(sa.String(250))
    datasource_type = sa.Column(sa.String(200))

def upgrade_slc(slc: Slice) -> None:
    if False:
        i = 10
        return i + 15
    slc.datasource_type = 'table'
    ds_id = None
    ds_type = None
    try:
        params_dict = json.loads(slc.params)
        (ds_id, ds_type) = params_dict['datasource'].split('__')
        params_dict['datasource'] = f'{ds_id}__table'
        slc.params = json.dumps(params_dict)
        logger.warning('updated slice datasource from %s__%s to %s__table for slice: %s', ds_id, ds_type, ds_id, slc.id)
    except Exception:
        logger.warning('failed to update slice.id = %s w/ datasource = %s__%s to %s__table', slc.id, ds_id, ds_type, ds_id)
        pass

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    with op.batch_alter_table('slices') as batch_op:
        for slc in session.query(Slice).filter(Slice.datasource_type != 'table').all():
            if slc.datasource_type == 'query':
                upgrade_slc(slc)
                session.add(slc)
            else:
                logger.warning('unknown value detected for slc.datasource_type: %s', slc.datasource_type)
    session.commit()
    with op.batch_alter_table('slices') as batch_op:
        batch_op.create_check_constraint('ck_chart_datasource', "datasource_type in ('table')")
    session.commit()
    session.close()

def downgrade():
    if False:
        return 10
    op.drop_constraint('ck_chart_datasource', 'slices', type_='check')