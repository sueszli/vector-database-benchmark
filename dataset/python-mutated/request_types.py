from dataclasses import asdict, dataclass
from typing import Any, Optional
from deepdiff import Delta

@dataclass
class _BaseRequest:

    def to_dict(self):
        if False:
            while True:
                i = 10
        return asdict(self)

@dataclass
class _DeltaRequest(_BaseRequest):
    delta: Delta

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        return self.delta.to_dict()

@dataclass
class _CommandRequest(_BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any

@dataclass
class _APIRequest(_BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any

@dataclass
class _RequestResponse(_BaseRequest):
    status_code: int
    content: Optional[str] = None