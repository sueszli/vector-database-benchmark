from dataclasses import dataclass
from azure.ai.ml.entities import EndpointAuthKeys

@dataclass
class DeploymentKeys:
    primary_key: str
    secondary_key: str

    @classmethod
    def _from_v2_endpoint_keys(cls, keys: EndpointAuthKeys) -> 'DeploymentKeys':
        if False:
            while True:
                i = 10
        return cls(primary_key=keys.primary_key, secondary_key=keys.secondary_key)