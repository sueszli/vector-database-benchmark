from ..test_config_base import TestConfigBase, NodeId

class TestConfig(TestConfigBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.nodes[NodeId.provider].opts = {'min_price': 0}
        self.nodes[NodeId.requestor].opts = {'max_price': 0}

    def update_task_dict(self):
        if False:
            return 10
        super().update_task_dict()
        self.task_dict['bid'] = 0