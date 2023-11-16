import json
from golem.config.active import EthereumConfig
from .modelbase import BasicModel

class NodeMetadataModel(BasicModel):

    def __init__(self, client, os_info, ver):
        if False:
            return 10
        super(NodeMetadataModel, self).__init__('NodeMetadata', client.get_key_id(), client.session_id)
        self.os_info = json.dumps({'type': 'OSInfo', 'obj': os_info.__dict__})
        self.settings = json.dumps({'type': 'ClientConfigDescriptor', 'obj': client.config_desc.__dict__})
        self.version = ver
        self.net = EthereumConfig().ACTIVE_NET

class NodeInfoModel(BasicModel):

    def __init__(self, cliid, sessid):
        if False:
            while True:
                i = 10
        super(NodeInfoModel, self).__init__('NodeInfo', cliid, sessid)