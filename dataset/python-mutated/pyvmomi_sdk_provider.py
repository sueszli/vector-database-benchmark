import atexit
import ssl
from pyVim.connect import Disconnect, SmartStubAdapter, VimSessionOrientedStub
from pyVmomi import vim
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client

@singleton_client
class PyvmomiSdkProvider:

    def __init__(self, server, user, password, session_type: Constants.SessionType, port: int=443):
        if False:
            for i in range(10):
                print('nop')
        self.server = server
        self.user = user
        self.password = password
        self.session_type = session_type
        self.port = port
        self.timeout = 0
        self.pyvmomi_sdk_client = self.get_client()
        if not self.pyvmomi_sdk_client:
            raise ValueError('Could not connect to the specified host')
        atexit.register(Disconnect, self.pyvmomi_sdk_client)

    def get_client(self):
        if False:
            i = 10
            return i + 15
        if self.session_type == Constants.SessionType.UNVERIFIED:
            context_obj = ssl._create_unverified_context()
        else:
            pass
        credentials = VimSessionOrientedStub.makeUserLoginMethod(self.user, self.password)
        smart_stub = SmartStubAdapter(host=self.server, port=self.port, sslContext=context_obj, connectionPoolTimeout=self.timeout)
        session_stub = VimSessionOrientedStub(smart_stub, credentials)
        return vim.ServiceInstance('ServiceInstance', session_stub)

    def get_pyvmomi_obj(self, vimtype, name=None, obj_id=None):
        if False:
            print('Hello World!')
        '\n        This function will return the vSphere object.\n        The argument for `vimtype` can be "vim.VM", "vim.Host", "vim.Datastore", etc.\n        Then either the name or the object id need to be provided.\n        To check all such object information, you can go to the managed object board\n        page of your vCenter Server, such as: https://<your_vc_ip/mob\n        '
        if not name and (not obj_id):
            raise RuntimeError('Either name or obj id must be provided')
        if self.pyvmomi_sdk_client is None:
            raise RuntimeError('Must init pyvmomi_sdk_client first')
        container = self.pyvmomi_sdk_client.content.viewManager.CreateContainerView(self.pyvmomi_sdk_client.content.rootFolder, vimtype, True)
        if name:
            for c in container.view:
                if c.name == name:
                    return c
        elif obj_id:
            for c in container.view:
                if obj_id in str(c):
                    return c
        raise ValueError(f'Cannot find the object with type {vimtype} on vSphere withname={name} and obj_id={obj_id}')

    def ensure_connect(self):
        if False:
            print('Hello World!')
        try:
            _ = self.pyvmomi_sdk_client.RetrieveContent()
        except vim.fault.NotAuthenticated:
            self.pyvmomi_sdk_client = self.get_client()
        except Exception as e:
            raise RuntimeError(f'failed to ensure the connect, exception: {e}')