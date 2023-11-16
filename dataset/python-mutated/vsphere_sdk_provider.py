import requests
from com.vmware.vapi.std.errors_client import Unauthenticated
from vmware.vapi.vsphere.client import create_vsphere_client
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client

def get_unverified_session():
    if False:
        for i in range(10):
            print('nop')
    '\n    vCenter provisioned internally have SSH certificates\n    expired so we use unverified session. Find out what\n    could be done for production.\n\n    Get a requests session with cert verification disabled.\n    Also disable the insecure warnings message.\n    Note this is not recommended in production code.\n    @return: a requests session with verification disabled.\n    '
    session = requests.session()
    session.verify = False
    requests.packages.urllib3.disable_warnings()
    return session

@singleton_client
class VsphereSdkProvider:

    def __init__(self, server, user, password, session_type: Constants.SessionType):
        if False:
            while True:
                i = 10
        self.server = server
        self.user = user
        self.password = password
        self.session_type = session_type
        self.vsphere_sdk_client = self.get_client()

    def get_client(self):
        if False:
            return 10
        session = None
        if self.session_type == Constants.SessionType.UNVERIFIED:
            session = get_unverified_session()
        else:
            pass
        return create_vsphere_client(server=self.server, username=self.user, password=self.password, session=session)

    def ensure_connect(self):
        if False:
            while True:
                i = 10
        try:
            _ = self.vsphere_sdk_client.vcenter.Cluster.list()
        except Unauthenticated:
            self.vsphere_sdk_client = self.get_client()
        except Exception as e:
            raise RuntimeError(f'failed to ensure the connect, exception: {e}')