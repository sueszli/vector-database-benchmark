import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from time import sleep
from typing import AbstractSet, Any, Mapping, Optional, Union
from typing_extensions import TypedDict
_DUMMY_VALUE = 100
_SOURCE_ASSETS: Mapping[str, Any] = {'delta': _DUMMY_VALUE}

class AssetSpec(TypedDict):
    key: str
    code_version: str
    dependencies: AbstractSet[str]

class SourceAssetSpec(TypedDict):
    key: str

class ProvenanceSpec(TypedDict):
    code_version: str
    input_data_versions: Mapping[str, str]

class MaterializeResult(TypedDict):
    data_version: str
    is_memoized: bool

class ObserveResult(TypedDict):
    data_version: str

class ExternalSystem:

    def __init__(self, storage_path: str):
        if False:
            print('Hello World!')
        self._db = _Database(storage_path)

    def materialize(self, asset_spec: AssetSpec, provenance_spec: Optional[ProvenanceSpec]) -> MaterializeResult:
        if False:
            print('Hello World!')
        'Recompute an asset if its provenance is missing or stale.\n\n        Receives asset provenance info from Dagster representing the last materialization on record.\n        The provenance is compared to the specified code version and current data versions of\n        dependencies to determine whether something has changed and the asset should be recomputed.\n\n        Args:\n            asset_spec (AssetSpec):\n                A dictionary containing an asset key, code version, and data dependencies.\n            provenance_spec (ProvenanceSpec):\n                A dictionary containing provenance info for the last materialization of the\n                specified asset. `None` if there is no materialization on record for the asset.\n\n        Returns (MaterializeResult):\n            A dictionary containing the data version for the asset and a boolean flag indicating\n            whether the data version corresponds to a memoized value (true) or a freshly computed\n            value (false).\n        '
        key = asset_spec['key']
        if not self._db.has(key) or provenance_spec is None or self._is_provenance_stale(asset_spec, provenance_spec):
            inputs = {dep: self._db.get(dep).value for dep in asset_spec['dependencies']}
            value = _compute_value(key, inputs, asset_spec['code_version'])
            data_version = _get_hash(value)
            record = _DatabaseRecord(value, data_version)
            self._db.set(key, record)
            is_memoized = False
        else:
            record = self._db.get(key)
            is_memoized = True
        return {'data_version': record.data_version, 'is_memoized': is_memoized}

    def observe(self, asset_spec: Union[AssetSpec, SourceAssetSpec]) -> ObserveResult:
        if False:
            for i in range(10):
                print('nop')
        'Observe an asset or source asset, returning its current data version.\n\n        Args:\n            asset_spec (Union[AssetSpec, SourceAssetSpec]):\n                A dictionary containing an asset key.\n\n        Returns (ObserveResult):\n            A dictionary containing the current data version of the asset.\n        '
        return {'data_version': self._db.get(asset_spec['key']).data_version}

    def _is_provenance_stale(self, asset_spec: AssetSpec, provenance_spec: ProvenanceSpec) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if provenance_spec['code_version'] != asset_spec['code_version']:
            return True
        if set(provenance_spec['input_data_versions'].keys()) != asset_spec['dependencies']:
            return True
        for (dep_key, version) in provenance_spec['input_data_versions'].items():
            if self._db.get(dep_key).data_version != version:
                return True
        return False

def _compute_value(key: str, inputs: Mapping[str, Any], code_version: str) -> Any:
    if False:
        i = 10
        return i + 15
    if code_version != 'lib/v1':
        raise Exception(f'Unknown code version {code_version}. Cannot compute.')
    if key == 'alpha':
        return 1
    elif key == 'beta':
        sleep(10)
        value = inputs['alpha'] + 1
        return value
    elif key == 'epsilon':
        sleep(10)
        return inputs['delta'] * 5

def _get_hash(value: Any) -> str:
    if False:
        while True:
            i = 10
    hash_sig = sha256()
    hash_sig.update(bytearray(str(value), 'utf8'))
    return hash_sig.hexdigest()[:6]

@dataclass
class _DatabaseRecord:
    value: Any
    data_version: str

class _Database:

    def __init__(self, storage_path: str):
        if False:
            print('Hello World!')
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)
        for (k, v) in _SOURCE_ASSETS.items():
            path = self.asset_path(k)
            if not os.path.exists(path):
                with open(self.asset_path(k), 'w') as fd:
                    record = _DatabaseRecord(v, _get_hash(v))
                    fd.write(json.dumps(asdict(record)))

    def asset_path(self, key: str) -> str:
        if False:
            return 10
        return f'{self.storage_path}/{key}.json'

    def get(self, key: str) -> _DatabaseRecord:
        if False:
            print('Hello World!')
        with open(self.asset_path(key), 'r') as fd:
            return _DatabaseRecord(**json.load(fd))

    def has(self, key: str) -> bool:
        if False:
            while True:
                i = 10
        return os.path.exists(self.asset_path(key))

    def set(self, key: str, record: _DatabaseRecord) -> None:
        if False:
            print('Hello World!')
        with open(self.asset_path(key), 'w') as fd:
            fd.write(json.dumps(asdict(record)))