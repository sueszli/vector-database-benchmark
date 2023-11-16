import json
from datetime import datetime
from typing import List, Optional
from lightning_cloud.openapi import Externalv1LightningappInstance, Externalv1Lightningwork, V1LightningappInstanceState, V1LightningappInstanceStatus
from rich.console import Console
from rich.table import Table
from rich.text import Text
from lightning.app.cli.core import Formatable
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient

class _AppManager:
    """_AppManager implements API calls specific to Lightning AI BYOC apps."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.api_client = LightningClient(retry=False)

    def get_app(self, app_id: str) -> Externalv1LightningappInstance:
        if False:
            for i in range(10):
                print('nop')
        project = _get_project(self.api_client)
        return self.api_client.lightningapp_instance_service_get_lightningapp_instance(project_id=project.project_id, id=app_id)

    def list_apps(self, limit: int=100, phase_in: Optional[List[str]]=None) -> List[Externalv1LightningappInstance]:
        if False:
            for i in range(10):
                print('nop')
        phase_in = phase_in or []
        project = _get_project(self.api_client)
        kwargs = {'project_id': project.project_id, 'limit': limit, 'phase_in': phase_in}
        resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**kwargs)
        apps = resp.lightningapps
        while resp.next_page_token is not None and resp.next_page_token != '':
            kwargs['page_token'] = resp.next_page_token
            resp = self.api_client.lightningapp_instance_service_list_lightningapp_instances(**kwargs)
            apps = apps + resp.lightningapps
        return apps

    def list_components(self, app_id: str, phase_in: Optional[List[str]]=None) -> List[Externalv1Lightningwork]:
        if False:
            print('Hello World!')
        phase_in = phase_in or []
        project = _get_project(self.api_client)
        resp = self.api_client.lightningwork_service_list_lightningwork(project_id=project.project_id, app_id=app_id, phase_in=phase_in)
        return resp.lightningworks

    def list(self, limit: int=100) -> None:
        if False:
            while True:
                i = 10
        console = Console()
        console.print(_AppList(self.list_apps(limit=limit)).as_table())

    def delete(self, app_id: str) -> None:
        if False:
            while True:
                i = 10
        project = _get_project(self.api_client)
        self.api_client.lightningapp_instance_service_delete_lightningapp_instance(project_id=project.project_id, id=app_id)

class _AppList(Formatable):

    def __init__(self, apps: List[Externalv1LightningappInstance]) -> None:
        if False:
            i = 10
            return i + 15
        self.apps = apps

    @staticmethod
    def _textualize_state_transitions(desired_state: V1LightningappInstanceState, current_state: V1LightningappInstanceStatus) -> Text:
        if False:
            while True:
                i = 10
        phases = {V1LightningappInstanceState.IMAGE_BUILDING: Text('building image', style='bold yellow'), V1LightningappInstanceState.PENDING: Text('pending', style='bold yellow'), V1LightningappInstanceState.RUNNING: Text('running', style='bold green'), V1LightningappInstanceState.FAILED: Text('failed', style='bold red'), V1LightningappInstanceState.STOPPED: Text('stopped'), V1LightningappInstanceState.NOT_STARTED: Text('not started'), V1LightningappInstanceState.DELETED: Text('deleted', style='bold red'), V1LightningappInstanceState.UNSPECIFIED: Text('unspecified', style='bold red')}
        if current_state.phase == V1LightningappInstanceState.UNSPECIFIED and current_state.start_timestamp is None:
            return Text('not yet started', style='bold yellow')
        if desired_state == V1LightningappInstanceState.DELETED and current_state.phase != V1LightningappInstanceState.DELETED:
            return Text('terminating', style='bold red')
        if any((phase == current_state.phase for phase in [V1LightningappInstanceState.PENDING, V1LightningappInstanceState.STOPPED])) and desired_state == V1LightningappInstanceState.RUNNING:
            return Text('restarting', style='bold yellow')
        return phases[current_state.phase]

    def as_json(self) -> str:
        if False:
            return 10
        return json.dumps(self.apps)

    def as_table(self) -> Table:
        if False:
            while True:
                i = 10
        table = Table('id', 'name', 'status', 'created', show_header=True, header_style='bold green')
        for app in self.apps:
            status = self._textualize_state_transitions(desired_state=app.spec.desired_state, current_state=app.status)
            created_at = datetime.now()
            if hasattr(app, 'created_at'):
                created_at = app.created_at
            table.add_row(app.id, app.name, status, created_at.strftime('%Y-%m-%d') if created_at else '')
        return table