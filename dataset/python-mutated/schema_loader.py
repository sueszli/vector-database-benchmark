from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class SchemaLoader:
    """Describes a stream's schema"""

    @abstractmethod
    def get_json_schema(self) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        "Returns a mapping describing the stream's schema"
        pass