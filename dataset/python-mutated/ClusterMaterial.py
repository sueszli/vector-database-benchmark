from ..BaseModel import BaseModel

class ClusterMaterial(BaseModel):

    def __init__(self, guid: str, version: int, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self.guid = guid
        self.version = version
        super().__init__(**kwargs)

    def validate(self) -> None:
        if False:
            return 10
        if not self.guid:
            raise ValueError('guid is required on ClusterMaterial')
        if not self.version:
            raise ValueError('version is required on ClusterMaterial')