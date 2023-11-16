from abc import ABC
from feast.repo_config import FeastConfigBaseModel

class OnlineStoreCreator(ABC):

    def __init__(self, project_name: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.project_name = project_name

    def create_online_store(self) -> FeastConfigBaseModel:
        if False:
            print('Hello World!')
        ...

    def teardown(self):
        if False:
            while True:
                i = 10
        ...