from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

class AbstractSchemaValidationPolicy(ABC):
    name: str
    validate_schema_before_sync = False

    @abstractmethod
    def record_passes_validation_policy(self, record: Mapping[str, Any], schema: Optional[Mapping[str, Any]]) -> bool:
        if False:
            return 10
        "\n        Return True if the record passes the user's validation policy.\n        "
        raise NotImplementedError()