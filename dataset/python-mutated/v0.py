import json
import logging
import time
from copy import copy
from datetime import datetime
from typing import Any, Optional
from flask_babel import lazy_gettext as _
from sqlalchemy.orm import make_transient, Session
from superset import db
from superset.commands.base import BaseCommand
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.datasets.commands.importers.v0 import import_dataset
from superset.exceptions import DashboardImportException
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.utils.dashboard_filter_scopes_converter import convert_filter_scopes, copy_filter_scopes
logger = logging.getLogger(__name__)

def import_chart(slc_to_import: Slice, slc_to_override: Optional[Slice], import_time: Optional[int]=None) -> int:
    if False:
        return 10
    'Inserts or overrides slc in the database.\n\n    remote_id and import_time fields in params_dict are set to track the\n    slice origin and ensure correct overrides for multiple imports.\n    Slice.perm is used to find the datasources and connect them.\n\n    :param Slice slc_to_import: Slice object to import\n    :param Slice slc_to_override: Slice to replace, id matches remote_id\n    :returns: The resulting id for the imported slice\n    :rtype: int\n    '
    session = db.session
    make_transient(slc_to_import)
    slc_to_import.dashboards = []
    slc_to_import.alter_params(remote_id=slc_to_import.id, import_time=import_time)
    slc_to_import = slc_to_import.copy()
    slc_to_import.reset_ownership()
    params = slc_to_import.params_dict
    datasource = SqlaTable.get_datasource_by_name(session=session, datasource_name=params['datasource_name'], database_name=params['database_name'], schema=params['schema'])
    slc_to_import.datasource_id = datasource.id
    if slc_to_override:
        slc_to_override.override(slc_to_import)
        session.flush()
        return slc_to_override.id
    session.add(slc_to_import)
    logger.info('Final slice: %s', str(slc_to_import.to_json()))
    session.flush()
    return slc_to_import.id

def import_dashboard(dashboard_to_import: Dashboard, dataset_id_mapping: Optional[dict[int, int]]=None, import_time: Optional[int]=None) -> int:
    if False:
        return 10
    "Imports the dashboard from the object to the database.\n\n    Once dashboard is imported, json_metadata field is extended and stores\n    remote_id and import_time. It helps to decide if the dashboard has to\n    be overridden or just copies over. Slices that belong to this\n    dashboard will be wired to existing tables. This function can be used\n    to import/export dashboards between multiple superset instances.\n    Audit metadata isn't copied over.\n    "

    def alter_positions(dashboard: Dashboard, old_to_new_slc_id_dict: dict[int, int]) -> None:
        if False:
            print('Hello World!')
        'Updates slice_ids in the position json.\n\n        Sample position_json data:\n        {\n            "DASHBOARD_VERSION_KEY": "v2",\n            "DASHBOARD_ROOT_ID": {\n                "type": "DASHBOARD_ROOT_TYPE",\n                "id": "DASHBOARD_ROOT_ID",\n                "children": ["DASHBOARD_GRID_ID"]\n            },\n            "DASHBOARD_GRID_ID": {\n                "type": "DASHBOARD_GRID_TYPE",\n                "id": "DASHBOARD_GRID_ID",\n                "children": ["DASHBOARD_CHART_TYPE-2"]\n            },\n            "DASHBOARD_CHART_TYPE-2": {\n                "type": "CHART",\n                "id": "DASHBOARD_CHART_TYPE-2",\n                "children": [],\n                "meta": {\n                    "width": 4,\n                    "height": 50,\n                    "chartId": 118\n                }\n            },\n        }\n        '
        position_data = json.loads(dashboard.position_json)
        position_json = position_data.values()
        for value in position_json:
            if isinstance(value, dict) and value.get('meta') and value.get('meta', {}).get('chartId'):
                old_slice_id = value['meta']['chartId']
                if old_slice_id in old_to_new_slc_id_dict:
                    value['meta']['chartId'] = old_to_new_slc_id_dict[old_slice_id]
        dashboard.position_json = json.dumps(position_data)

    def alter_native_filters(dashboard: Dashboard) -> None:
        if False:
            print('Hello World!')
        json_metadata = json.loads(dashboard.json_metadata)
        native_filter_configuration = json_metadata.get('native_filter_configuration')
        if not native_filter_configuration:
            return
        for native_filter in native_filter_configuration:
            for target in native_filter.get('targets', []):
                old_dataset_id = target.get('datasetId')
                if dataset_id_mapping and old_dataset_id is not None:
                    target['datasetId'] = dataset_id_mapping.get(old_dataset_id, old_dataset_id)
        dashboard.json_metadata = json.dumps(json_metadata)
    logger.info('Started import of the dashboard: %s', dashboard_to_import.to_json())
    session = db.session
    logger.info('Dashboard has %d slices', len(dashboard_to_import.slices))
    slices = copy(dashboard_to_import.slices)
    dashboard_to_import.slug = None
    old_json_metadata = json.loads(dashboard_to_import.json_metadata or '{}')
    old_to_new_slc_id_dict: dict[int, int] = {}
    new_timed_refresh_immune_slices = []
    new_expanded_slices = {}
    new_filter_scopes = {}
    i_params_dict = dashboard_to_import.params_dict
    remote_id_slice_map = {slc.params_dict['remote_id']: slc for slc in session.query(Slice).all() if 'remote_id' in slc.params_dict}
    for slc in slices:
        logger.info('Importing slice %s from the dashboard: %s', slc.to_json(), dashboard_to_import.dashboard_title)
        remote_slc = remote_id_slice_map.get(slc.id)
        new_slc_id = import_chart(slc, remote_slc, import_time=import_time)
        old_to_new_slc_id_dict[slc.id] = new_slc_id
        new_slc_id_str = str(new_slc_id)
        old_slc_id_str = str(slc.id)
        if 'timed_refresh_immune_slices' in i_params_dict and old_slc_id_str in i_params_dict['timed_refresh_immune_slices']:
            new_timed_refresh_immune_slices.append(new_slc_id_str)
        if 'expanded_slices' in i_params_dict and old_slc_id_str in i_params_dict['expanded_slices']:
            new_expanded_slices[new_slc_id_str] = i_params_dict['expanded_slices'][old_slc_id_str]
    filter_scopes = {}
    if 'filter_immune_slices' in i_params_dict or 'filter_immune_slice_fields' in i_params_dict:
        filter_scopes = convert_filter_scopes(old_json_metadata, slices)
    if 'filter_scopes' in i_params_dict:
        filter_scopes = old_json_metadata.get('filter_scopes')
    if filter_scopes:
        new_filter_scopes = copy_filter_scopes(old_to_new_slc_id_dict=old_to_new_slc_id_dict, old_filter_scopes=filter_scopes)
    existing_dashboard = None
    for dash in session.query(Dashboard).all():
        if 'remote_id' in dash.params_dict and dash.params_dict['remote_id'] == dashboard_to_import.id:
            existing_dashboard = dash
    dashboard_to_import = dashboard_to_import.copy()
    dashboard_to_import.id = None
    dashboard_to_import.reset_ownership()
    if dashboard_to_import.position_json:
        alter_positions(dashboard_to_import, old_to_new_slc_id_dict)
    dashboard_to_import.alter_params(import_time=import_time)
    dashboard_to_import.remove_params(param_to_remove='filter_immune_slices')
    dashboard_to_import.remove_params(param_to_remove='filter_immune_slice_fields')
    if new_filter_scopes:
        dashboard_to_import.alter_params(filter_scopes=new_filter_scopes)
    if new_expanded_slices:
        dashboard_to_import.alter_params(expanded_slices=new_expanded_slices)
    if new_timed_refresh_immune_slices:
        dashboard_to_import.alter_params(timed_refresh_immune_slices=new_timed_refresh_immune_slices)
    alter_native_filters(dashboard_to_import)
    new_slices = session.query(Slice).filter(Slice.id.in_(old_to_new_slc_id_dict.values())).all()
    if existing_dashboard:
        existing_dashboard.override(dashboard_to_import)
        existing_dashboard.slices = new_slices
        session.flush()
        return existing_dashboard.id
    dashboard_to_import.slices = new_slices
    session.add(dashboard_to_import)
    session.flush()
    return dashboard_to_import.id

def decode_dashboards(o: dict[str, Any]) -> Any:
    if False:
        print('Hello World!')
    '\n    Function to be passed into json.loads obj_hook parameter\n    Recreates the dashboard object from a json representation.\n    '
    if '__Dashboard__' in o:
        return Dashboard(**o['__Dashboard__'])
    if '__Slice__' in o:
        return Slice(**o['__Slice__'])
    if '__TableColumn__' in o:
        return TableColumn(**o['__TableColumn__'])
    if '__SqlaTable__' in o:
        return SqlaTable(**o['__SqlaTable__'])
    if '__SqlMetric__' in o:
        return SqlMetric(**o['__SqlMetric__'])
    if '__datetime__' in o:
        return datetime.strptime(o['__datetime__'], '%Y-%m-%dT%H:%M:%S')
    return o

def import_dashboards(session: Session, content: str, database_id: Optional[int]=None, import_time: Optional[int]=None) -> None:
    if False:
        while True:
            i = 10
    'Imports dashboards from a stream to databases'
    current_tt = int(time.time())
    import_time = current_tt if import_time is None else import_time
    data = json.loads(content, object_hook=decode_dashboards)
    if not data:
        raise DashboardImportException(_('No data in file'))
    dataset_id_mapping: dict[int, int] = {}
    for table in data['datasources']:
        new_dataset_id = import_dataset(table, database_id, import_time=import_time)
        params = json.loads(table.params)
        dataset_id_mapping[params['remote_id']] = new_dataset_id
    session.commit()
    for dashboard in data['dashboards']:
        import_dashboard(dashboard, dataset_id_mapping, import_time=import_time)
    session.commit()

class ImportDashboardsCommand(BaseCommand):
    """
    Import dashboard in JSON format.

    This is the original unversioned format used to export and import dashboards
    in Superset.
    """

    def __init__(self, contents: dict[str, str], database_id: Optional[int]=None, **kwargs: Any):
        if False:
            print('Hello World!')
        self.contents = contents
        self.database_id = database_id

    def run(self) -> None:
        if False:
            print('Hello World!')
        self.validate()
        for (file_name, content) in self.contents.items():
            logger.info('Importing dashboard from file %s', file_name)
            import_dashboards(db.session, content, self.database_id)

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        for content in self.contents.values():
            try:
                json.loads(content)
            except ValueError:
                logger.exception('Invalid JSON file')
                raise