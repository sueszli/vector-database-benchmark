"""cross-filter-global-scoping

Revision ID: 4ea966691069
Revises: 7e67aecbf3f1
Create Date: 2023-05-11 12:41:38.095717

"""
revision = '4ea966691069'
down_revision = '7e67aecbf3f1'
import copy
import json
import logging
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from superset import db
from superset.migrations.shared.utils import paginated_update
Base = declarative_base()
logger = logging.getLogger(__name__)

class Dashboard(Base):
    __tablename__ = 'dashboards'
    id = sa.Column(sa.Integer, primary_key=True)
    json_metadata = sa.Column(sa.Text)

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for dashboard in paginated_update(session.query(Dashboard)):
        needs_upgrade = True
        try:
            json_metadata = json.loads(dashboard.json_metadata or '{}')
            new_chart_configuration = {}
            for config in json_metadata.get('chart_configuration', {}).values():
                if not isinstance(config, dict):
                    continue
                chart_id = int(config.get('id', 0))
                scope = config.get('crossFilters', {}).get('scope', {})
                if not isinstance(scope, dict):
                    needs_upgrade = False
                    continue
                excluded = [int(excluded_id) for excluded_id in scope.get('excluded', [])]
                new_chart_configuration[chart_id] = copy.deepcopy(config)
                new_chart_configuration[chart_id]['id'] = chart_id
                new_chart_configuration[chart_id]['crossFilters']['scope']['excluded'] = excluded
                if scope.get('rootPath') == ['ROOT_ID'] and excluded == [chart_id]:
                    new_chart_configuration[chart_id]['crossFilters']['scope'] = 'global'
            json_metadata['chart_configuration'] = new_chart_configuration
            if needs_upgrade:
                dashboard.json_metadata = json.dumps(json_metadata)
        except Exception as e:
            logger.exception('Failed to run up migration')
            raise e
    session.commit()
    session.close()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for dashboard in paginated_update(session.query(Dashboard)):
        try:
            json_metadata = json.loads(dashboard.json_metadata)
            new_chart_configuration = {}
            for config in json_metadata.get('chart_configuration', {}).values():
                if not isinstance(config, dict):
                    continue
                chart_id = config.get('id')
                if chart_id is None:
                    continue
                scope = config.get('crossFilters', {}).get('scope', {})
                new_chart_configuration[chart_id] = copy.deepcopy(config)
                if scope in ('global', 'Global'):
                    new_chart_configuration[chart_id]['crossFilters']['scope'] = {'rootPath': ['ROOT_ID'], 'excluded': [chart_id]}
            json_metadata['chart_configuration'] = new_chart_configuration
            if 'global_chart_configuration' in json_metadata:
                del json_metadata['global_chart_configuration']
            dashboard.json_metadata = json.dumps(json_metadata)
        except Exception as e:
            logger.exception('Failed to run down migration')
            raise e
    session.commit()
    session.close()