import copy
from pytest_mock import MockFixture
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import select
from tests.unit_tests.fixtures.assets_configs import charts_config_1, charts_config_2, dashboards_config_1, dashboards_config_2, databases_config, datasets_config

def test_import_new_assets(mocker: MockFixture, session: Session) -> None:
    if False:
        return 10
    '\n    Test that all new assets are imported correctly.\n    '
    from superset import security_manager
    from superset.commands.importers.v1.assets import ImportAssetsCommand
    from superset.models.dashboard import dashboard_slices
    from superset.models.slice import Slice
    mocker.patch.object(security_manager, 'can_access', return_value=True)
    engine = session.get_bind()
    Slice.metadata.create_all(engine)
    configs = {**copy.deepcopy(databases_config), **copy.deepcopy(datasets_config), **copy.deepcopy(charts_config_1), **copy.deepcopy(dashboards_config_1)}
    expected_number_of_dashboards = len(dashboards_config_1)
    expected_number_of_charts = len(charts_config_1)
    ImportAssetsCommand._import(session, configs)
    dashboard_ids = session.scalars(select(dashboard_slices.c.dashboard_id).distinct()).all()
    chart_ids = session.scalars(select(dashboard_slices.c.slice_id)).all()
    assert len(chart_ids) == expected_number_of_charts
    assert len(dashboard_ids) == expected_number_of_dashboards

def test_import_adds_dashboard_charts(mocker: MockFixture, session: Session) -> None:
    if False:
        print('Hello World!')
    '\n    Test that existing dashboards are updated with new charts.\n    '
    from superset import security_manager
    from superset.commands.importers.v1.assets import ImportAssetsCommand
    from superset.models.dashboard import dashboard_slices
    from superset.models.slice import Slice
    mocker.patch.object(security_manager, 'can_access', return_value=True)
    engine = session.get_bind()
    Slice.metadata.create_all(engine)
    base_configs = {**copy.deepcopy(databases_config), **copy.deepcopy(datasets_config), **copy.deepcopy(charts_config_2), **copy.deepcopy(dashboards_config_2)}
    new_configs = {**copy.deepcopy(databases_config), **copy.deepcopy(datasets_config), **copy.deepcopy(charts_config_1), **copy.deepcopy(dashboards_config_1)}
    expected_number_of_dashboards = len(dashboards_config_1)
    expected_number_of_charts = len(charts_config_1)
    ImportAssetsCommand._import(session, base_configs)
    ImportAssetsCommand._import(session, new_configs)
    dashboard_ids = session.scalars(select(dashboard_slices.c.dashboard_id).distinct()).all()
    chart_ids = session.scalars(select(dashboard_slices.c.slice_id)).all()
    assert len(chart_ids) == expected_number_of_charts
    assert len(dashboard_ids) == expected_number_of_dashboards

def test_import_removes_dashboard_charts(mocker: MockFixture, session: Session) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that existing dashboards are updated without old charts.\n    '
    from superset import security_manager
    from superset.commands.importers.v1.assets import ImportAssetsCommand
    from superset.models.dashboard import dashboard_slices
    from superset.models.slice import Slice
    mocker.patch.object(security_manager, 'can_access', return_value=True)
    engine = session.get_bind()
    Slice.metadata.create_all(engine)
    base_configs = {**copy.deepcopy(databases_config), **copy.deepcopy(datasets_config), **copy.deepcopy(charts_config_1), **copy.deepcopy(dashboards_config_1)}
    new_configs = {**copy.deepcopy(databases_config), **copy.deepcopy(datasets_config), **copy.deepcopy(charts_config_2), **copy.deepcopy(dashboards_config_2)}
    expected_number_of_dashboards = len(dashboards_config_2)
    expected_number_of_charts = len(charts_config_2)
    ImportAssetsCommand._import(session, base_configs)
    ImportAssetsCommand._import(session, new_configs)
    dashboard_ids = session.scalars(select(dashboard_slices.c.dashboard_id).distinct()).all()
    chart_ids = session.scalars(select(dashboard_slices.c.slice_id)).all()
    assert len(chart_ids) == expected_number_of_charts
    assert len(dashboard_ids) == expected_number_of_dashboards