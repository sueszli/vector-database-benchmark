from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING
from ..abstract_node import NodeSideType
from ..client.api import APIRegistry
from ..img.base64 import base64read
from ..serde.serializable import serializable
from ..service.network.routes import NodeRouteType
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.fonts import fonts_css
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest
if TYPE_CHECKING:
    from ..service.code.user_code import SubmitUserCode

@serializable()
class EnclaveMetadata(SyftObject):
    __canonical_name__ = 'EnclaveMetadata'
    __version__ = SYFT_OBJECT_VERSION_1
    route: NodeRouteType

@serializable()
class EnclaveClient(SyftClient):
    __api_patched = False

    @property
    def code(self) -> Optional[APIModule]:
        if False:
            i = 10
            return i + 15
        if self.api.has_service('code'):
            res = self.api.services.code
            if not self.__api_patched:
                self._request_code_execution = res.request_code_execution
                self.__api_patched = True
            res.request_code_execution = self.request_code_execution
            return res
        return None

    @property
    def requests(self) -> Optional[APIModule]:
        if False:
            for i in range(10):
                print('nop')
        if self.api.has_service('request'):
            return self.api.services.request
        return None

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

    def get_enclave_metadata(self) -> EnclaveMetadata:
        if False:
            for i in range(10):
                print('nop')
        return EnclaveMetadata(route=self.connection.route)

    def request_code_execution(self, code: SubmitUserCode):
        if False:
            return 10
        from ..service.code.user_code_service import SubmitUserCode
        if not isinstance(code, SubmitUserCode):
            raise Exception(f'The input code should be of type: {SubmitUserCode} got:{type(code)}')
        enclave_metadata = self.get_enclave_metadata()
        code_id = UID()
        code.id = code_id
        code.enclave_metadata = enclave_metadata
        apis = []
        for (k, v) in code.input_policy_init_kwargs.items():
            api = APIRegistry.get_by_recent_node_uid(k.node_id)
            if api is None:
                raise ValueError(f'could not find client for input {v}')
            else:
                apis += [api]
        for api in apis:
            res = api.services.code.request_code_execution(code=code)
            if isinstance(res, SyftError):
                return res
        _ = self.code
        res = self._request_code_execution(code=code)
        return res

    def _repr_html_(self) -> str:
        if False:
            print('Hello World!')
        commands = "\n        <li><span class='syft-code-block'>&lt;your_client&gt;\n        .request_code_execution</span> - submit code to enclave for execution</li>\n        "
        command_list = f"\n        <ul style='padding-left: 1em;'>\n            {commands}\n        </ul>\n        "
        small_grid_symbol_logo = base64read('small-grid-symbol-logo.png')
        url = getattr(self.connection, 'url', None)
        node_details = f'<strong>URL:</strong> {url}<br />' if url else ''
        node_details += f'<strong>Node Type:</strong> {self.metadata.node_type.capitalize()}<br />'
        node_side_type = 'Low Side' if self.metadata.node_side_type == NodeSideType.LOW_SIDE.value else 'High Side'
        node_details += f'<strong>Node Side Type:</strong> {node_side_type}<br />'
        node_details += f'<strong>Syft Version:</strong> {self.metadata.syft_version}<br />'
        return f'''\n        <style>\n            {fonts_css}\n\n            .syft-container {{\n                padding: 5px;\n                font-family: 'Open Sans';\n            }}\n            .syft-alert-info {{\n                color: #1F567A;\n                background-color: #C2DEF0;\n                border-radius: 4px;\n                padding: 5px;\n                padding: 13px 10px\n            }}\n            .syft-code-block {{\n                background-color: #f7f7f7;\n                border: 1px solid #cfcfcf;\n                padding: 0px 2px;\n            }}\n            .syft-space {{\n                margin-top: 1em;\n            }}\n        </style>\n        <div class="syft-client syft-container">\n            <img src="{small_grid_symbol_logo}" alt="Logo"\n            style="width:48px;height:48px;padding:3px;">\n            <h2>Welcome to {self.name}</h2>\n            <div class="syft-space">\n                {node_details}\n            </div>\n            <div class='syft-alert-info syft-space'>\n                &#9432;&nbsp;\n                This node is run by the library PySyft to learn more about how it works visit\n                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.\n            </div>\n            <h4>Commands to Get Started</h4>\n            {command_list}\n        </div><br />\n        '''