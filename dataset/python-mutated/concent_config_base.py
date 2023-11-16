from ..test_config_base import TestConfigBase, CONCENT_STAGING

class ConcentTestConfigBase(TestConfigBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.enable_concent()

    def enable_concent(self):
        if False:
            print('Hello World!')
        node_configs = []
        for n in self.nodes.values():
            node_configs.extend(n if isinstance(n, list) else [n])
        for node_config in node_configs:
            node_config.concent = CONCENT_STAGING