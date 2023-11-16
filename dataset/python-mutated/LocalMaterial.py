from .BaseModel import BaseModel

class LocalMaterial(BaseModel):

    def __init__(self, GUID: str, id: str, version: int, **kwargs) -> None:
        if False:
            return 10
        self.GUID = GUID
        self.id = id
        self.version = version
        super().__init__(**kwargs)

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().validate()
        if not self.GUID:
            raise ValueError('guid is required on LocalMaterial')
        if not self.version:
            raise ValueError('version is required on LocalMaterial')
        if not self.id:
            raise ValueError('id is required on LocalMaterial')