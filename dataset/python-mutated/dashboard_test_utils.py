import logging
import random
import string
from typing import Any, Optional
from sqlalchemy import func
from superset import appbuilder, db, security_manager
from superset.connectors.sqla.models import SqlaTable
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from tests.integration_tests.dashboards.consts import DEFAULT_DASHBOARD_SLUG_TO_TEST
logger = logging.getLogger(__name__)
session = appbuilder.get_session

def get_mock_positions(dashboard: Dashboard) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    positions = {'DASHBOARD_VERSION_KEY': 'v2'}
    for (i, slc) in enumerate(dashboard.slices):
        id_ = f'DASHBOARD_CHART_TYPE-{i}'
        position_data: Any = {'type': 'CHART', 'id': id_, 'children': [], 'meta': {'width': 4, 'height': 50, 'chartId': slc.id}}
        positions[id_] = position_data
    return positions

def build_save_dash_parts(dashboard_slug: Optional[str]=None, dashboard_to_edit: Optional[Dashboard]=None) -> tuple[Dashboard, dict[str, Any], dict[str, Any]]:
    if False:
        print('Hello World!')
    if not dashboard_to_edit:
        dashboard_slug = dashboard_slug if dashboard_slug else DEFAULT_DASHBOARD_SLUG_TO_TEST
        dashboard_to_edit = get_dashboard_by_slug(dashboard_slug)
    data_before_change = {'positions': dashboard_to_edit.position, 'dashboard_title': dashboard_to_edit.dashboard_title}
    data_after_change = {'css': '', 'expanded_slices': {}, 'positions': get_mock_positions(dashboard_to_edit), 'dashboard_title': dashboard_to_edit.dashboard_title}
    return (dashboard_to_edit, data_before_change, data_after_change)

def get_all_dashboards() -> list[Dashboard]:
    if False:
        for i in range(10):
            print('nop')
    return db.session.query(Dashboard).all()

def get_dashboard_by_slug(dashboard_slug: str) -> Dashboard:
    if False:
        while True:
            i = 10
    return db.session.query(Dashboard).filter_by(slug=dashboard_slug).first()

def get_slice_by_name(slice_name: str) -> Slice:
    if False:
        for i in range(10):
            print('nop')
    return db.session.query(Slice).filter_by(slice_name=slice_name).first()

def get_sql_table_by_name(table_name: str):
    if False:
        for i in range(10):
            print('nop')
    return db.session.query(SqlaTable).filter_by(table_name=table_name).one()

def count_dashboards() -> int:
    if False:
        for i in range(10):
            print('nop')
    return db.session.query(func.count(Dashboard.id)).first()[0]

def random_title():
    if False:
        while True:
            i = 10
    return f'title{random_str()}'

def random_slug():
    if False:
        for i in range(10):
            print('nop')
    return f'slug{random_str()}'

def get_random_string(length):
    if False:
        while True:
            i = 10
    letters = string.ascii_lowercase
    result_str = ''.join((random.choice(letters) for i in range(length)))
    print('Random string of length', length, 'is:', result_str)
    return result_str

def random_str():
    if False:
        return 10
    return get_random_string(8)

def grant_access_to_dashboard(dashboard, role_name):
    if False:
        return 10
    role = security_manager.find_role(role_name)
    dashboard.roles.append(role)
    db.session.merge(dashboard)
    db.session.commit()

def revoke_access_to_dashboard(dashboard, role_name):
    if False:
        return 10
    role = security_manager.find_role(role_name)
    dashboard.roles.remove(role)
    db.session.merge(dashboard)
    db.session.commit()