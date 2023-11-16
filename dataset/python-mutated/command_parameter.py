import dataclasses
from autogpt.core.utils.json_schema import JSONSchema

@dataclasses.dataclass
class CommandParameter:
    name: str
    spec: JSONSchema

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f"CommandParameter('{self.name}', '{self.spec.type}', '{self.spec.description}', {self.spec.required})"