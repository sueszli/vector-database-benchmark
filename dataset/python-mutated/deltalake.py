from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.utils.module_loading import qualname
serializers = ['deltalake.table.DeltaTable']
deserializers = serializers
stringifiers = serializers
if TYPE_CHECKING:
    from airflow.serialization.serde import U
__version__ = 1

def serialize(o: object) -> tuple[U, str, int, bool]:
    if False:
        return 10
    from deltalake.table import DeltaTable
    if not isinstance(o, DeltaTable):
        return ('', '', 0, False)
    from airflow.models.crypto import get_fernet
    fernet = get_fernet()
    properties: dict = {}
    for (k, v) in o._storage_options.items() if o._storage_options else {}:
        properties[k] = fernet.encrypt(v.encode('utf-8')).decode('utf-8')
    data = {'table_uri': o.table_uri, 'version': o.version(), 'storage_options': properties}
    return (data, qualname(o), __version__, True)

def deserialize(classname: str, version: int, data: dict):
    if False:
        for i in range(10):
            print('nop')
    from deltalake.table import DeltaTable
    from airflow.models.crypto import get_fernet
    if version > __version__:
        raise TypeError('serialized version is newer than class version')
    if classname == qualname(DeltaTable):
        fernet = get_fernet()
        properties = {}
        for (k, v) in data['storage_options'].items():
            properties[k] = fernet.decrypt(v.encode('utf-8')).decode('utf-8')
        if len(properties) == 0:
            storage_options = None
        else:
            storage_options = properties
        return DeltaTable(data['table_uri'], version=data['version'], storage_options=storage_options)
    raise TypeError(f'do not know how to deserialize {classname}')