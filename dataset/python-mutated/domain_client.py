from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from tqdm import tqdm
from ..abstract_node import NodeSideType
from ..img.base64 import base64read
from ..serde.serializable import serializable
from ..service.code_history.code_history import CodeHistoriesDict
from ..service.code_history.code_history import UsersCodeHistoriesDict
from ..service.dataset.dataset import Contributor
from ..service.dataset.dataset import CreateAsset
from ..service.dataset.dataset import CreateDataset
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.user.roles import Roles
from ..service.user.user_roles import ServiceRole
from ..types.uid import UID
from ..util.fonts import fonts_css
from ..util.util import get_mb_size
from ..util.util import prompt_warning_message
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest
if TYPE_CHECKING:
    from ..service.project.project import Project

def add_default_uploader(user, obj: Union[CreateDataset, CreateAsset]) -> Union[CreateDataset, CreateAsset]:
    if False:
        while True:
            i = 10
    uploader = None
    for contributor in obj.contributors:
        if contributor.role == str(Roles.UPLOADER):
            uploader = contributor
            break
    if uploader is None:
        uploader = Contributor(role=str(Roles.UPLOADER), name=user.name, email=user.email)
        obj.contributors.add(uploader)
    obj.uploader = uploader
    return obj

@serializable()
class DomainClient(SyftClient):

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'<DomainClient: {self.name}>'

    def upload_dataset(self, dataset: CreateDataset) -> Union[SyftSuccess, SyftError]:
        if False:
            i = 10
            return i + 15
        from ..types.twin_object import TwinObject
        user = self.users.get_current_user()
        dataset = add_default_uploader(user, dataset)
        for i in range(len(dataset.asset_list)):
            asset = dataset.asset_list[i]
            dataset.asset_list[i] = add_default_uploader(user, asset)
        dataset._check_asset_must_contain_mock()
        dataset_size = 0
        metadata = self.api.connection.get_node_metadata(self.api.signing_key)
        if metadata.show_warnings and metadata.node_side_type == NodeSideType.HIGH_SIDE.value:
            message = f"You're approving a request on {metadata.node_side_type} side {metadata.node_type} which may host datasets with private information."
            prompt_warning_message(message=message, confirm=True)
        for asset in tqdm(dataset.asset_list):
            print(f'Uploading: {asset.name}')
            try:
                twin = TwinObject(private_obj=asset.data, mock_obj=asset.mock, syft_node_location=self.id, syft_client_verify_key=self.verify_key)
                twin._save_to_blob_storage()
            except Exception as e:
                return SyftError(message=f'Failed to create twin. {e}')
            response = self.api.services.action.set(twin)
            if isinstance(response, SyftError):
                print(f'Failed to upload asset\n: {asset}')
                return response
            asset.action_id = twin.id
            asset.node_uid = self.id
            dataset_size += get_mb_size(asset.data)
        dataset.mb_size = dataset_size
        valid = dataset.check()
        if valid.ok():
            return self.api.services.dataset.add(dataset=dataset)
        else:
            if len(valid.err()) > 0:
                return tuple(valid.err())
            return valid.err()

    def connect_to_gateway(self, via_client: Optional[SyftClient]=None, url: Optional[str]=None, port: Optional[int]=None, handle: Optional[NodeHandle]=None, email: Optional[str]=None, password: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = login_as_guest(url=url, port=port) if email is None else login(url=url, port=port, email=email, password=password)
            if isinstance(client, SyftError):
                return client
        res = self.exchange_route(client)
        if isinstance(res, SyftSuccess):
            return SyftSuccess(message=f'Connected {self.metadata.node_type} to {client.name} gateway')
        return res

    @property
    def data_subject_registry(self) -> Optional[APIModule]:
        if False:
            return 10
        if self.api.has_service('data_subject'):
            return self.api.services.data_subject
        return None

    @property
    def code(self) -> Optional[APIModule]:
        if False:
            while True:
                i = 10
        if self.api.has_service('code'):
            return self.api.services.code
        return None

    @property
    def requests(self) -> Optional[APIModule]:
        if False:
            print('Hello World!')
        if self.api.has_service('request'):
            return self.api.services.request
        return None

    @property
    def datasets(self) -> Optional[APIModule]:
        if False:
            i = 10
            return i + 15
        if self.api.has_service('dataset'):
            return self.api.services.dataset
        return None

    @property
    def projects(self) -> Optional[APIModule]:
        if False:
            for i in range(10):
                print('nop')
        if self.api.has_service('project'):
            return self.api.services.project
        return None

    @property
    def code_history_service(self) -> Optional[APIModule]:
        if False:
            for i in range(10):
                print('nop')
        if self.api is not None and self.api.has_service('code_history'):
            return self.api.services.code_history
        return None

    @property
    def code_history(self) -> CodeHistoriesDict:
        if False:
            return 10
        return self.api.services.code_history.get_history()

    @property
    def code_histories(self) -> UsersCodeHistoriesDict:
        if False:
            return 10
        return self.api.services.code_history.get_histories()

    def get_project(self, name: str=None, uid: UID=None) -> Optional[Project]:
        if False:
            while True:
                i = 10
        'Get project by name or UID'
        if not self.api.has_service('project'):
            return None
        if name:
            return self.api.services.project.get_by_name(name)
        elif uid:
            return self.api.services.project.get_by_uid(uid)
        return self.api.services.project.get_all()

    def _repr_html_(self) -> str:
        if False:
            return 10
        guest_commands = "\n        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.login</span> - list projects</li>\n        <li>\n            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature\n        </li>"
        ds_commands = "\n        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>\n        <li>\n            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature\n        </li>"
        do_commands = "\n        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.requests</span> - list requests</li>\n        <li><span class='syft-code-block'>&lt;your_client&gt;.users</span> - list users</li>\n        <li>\n            <span class='syft-code-block'>&lt;your_client&gt;.requests.submit?</span> - display function signature\n        </li>"
        if self.user_role.value == ServiceRole.NONE.value or self.user_role.value == ServiceRole.GUEST.value:
            commands = guest_commands
        elif self.user_role is not None and self.user_role.value == ServiceRole.DATA_SCIENTIST.value:
            commands = ds_commands
        elif self.user_role is not None and self.user_role.value >= ServiceRole.DATA_OWNER.value:
            commands = do_commands
        command_list = f"\n        <ul style='padding-left: 1em;'>\n            {commands}\n        </ul>\n        "
        small_grid_symbol_logo = base64read('small-grid-symbol-logo.png')
        url = getattr(self.connection, 'url', None)
        node_details = f'<strong>URL:</strong> {url}<br />' if url else ''
        node_details += f'<strong>Node Type:</strong> {self.metadata.node_type.capitalize()}<br />'
        node_side_type = 'Low Side' if self.metadata.node_side_type == NodeSideType.LOW_SIDE.value else 'High Side'
        node_details += f'<strong>Node Side Type:</strong> {node_side_type}<br />'
        node_details += f'<strong>Syft Version:</strong> {self.metadata.syft_version}<br />'
        return f'''\n        <style>\n            {fonts_css}\n\n            .syft-container {{\n                padding: 5px;\n                font-family: 'Open Sans';\n            }}\n            .syft-alert-info {{\n                color: #1F567A;\n                background-color: #C2DEF0;\n                border-radius: 4px;\n                padding: 5px;\n                padding: 13px 10px\n            }}\n            .syft-code-block {{\n                background-color: #f7f7f7;\n                border: 1px solid #cfcfcf;\n                padding: 0px 2px;\n            }}\n            .syft-space {{\n                margin-top: 1em;\n            }}\n        </style>\n        <div class="syft-client syft-container">\n            <img src="{small_grid_symbol_logo}" alt="Logo"\n            style="width:48px;height:48px;padding:3px;">\n            <h2>Welcome to {self.name}</h2>\n            <div class="syft-space">\n                {node_details}\n            </div>\n            <div class='syft-alert-info syft-space'>\n                &#9432;&nbsp;\n                This domain is run by the library PySyft to learn more about how it works visit\n                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.\n            </div>\n            <h4>Commands to Get Started</h4>\n            {command_list}\n        </div><br />\n        '''