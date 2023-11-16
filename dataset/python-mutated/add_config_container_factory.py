"""check that factory for a container works"""
from __future__ import annotations
from typing import List
from tox.config.sets import ConfigSet

class EnvDockerConfigSet(ConfigSet):

    def register_config(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def factory(container_name: object) -> str:
            if False:
                return 10
            raise NotImplementedError
        self.add_config(keys=['k'], of_type=List[str], default=[], desc='desc', factory=factory)