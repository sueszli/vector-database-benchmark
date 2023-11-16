from ..BaseModel import BaseModel

class ClusterBuildPlate(BaseModel):
    """Class representing a cluster printer"""

    def __init__(self, type: str='glass', **kwargs) -> None:
        if False:
            print('Hello World!')
        'Create a new build plate\n\n        :param type: The type of build plate glass or aluminium\n        '
        self.type = type
        super().__init__(**kwargs)