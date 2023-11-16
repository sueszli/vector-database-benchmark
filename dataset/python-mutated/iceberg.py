from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.utils.module_loading import qualname
serializers = ['pyiceberg.table.Table']
deserializers = serializers
stringifiers = serializers
if TYPE_CHECKING:
    from airflow.serialization.serde import U
__version__ = 1

def serialize(o: object) -> tuple[U, str, int, bool]:
    if False:
        return 10
    from pyiceberg.table import Table
    if not isinstance(o, Table):
        return ('', '', 0, False)
    from airflow.models.crypto import get_fernet
    fernet = get_fernet()
    properties = {}
    for (k, v) in o.catalog.properties.items():
        properties[k] = fernet.encrypt(v.encode('utf-8')).decode('utf-8')
    data = {'identifier': o.identifier, 'catalog_properties': properties}
    return (data, qualname(o), __version__, True)

def deserialize(classname: str, version: int, data: dict):
    if False:
        print('Hello World!')
    from pyiceberg.catalog import load_catalog
    from pyiceberg.table import Table
    from airflow.models.crypto import get_fernet
    if version > __version__:
        raise TypeError('serialized version is newer than class version')
    if classname == qualname(Table):
        fernet = get_fernet()
        properties = {}
        for (k, v) in data['catalog_properties'].items():
            properties[k] = fernet.decrypt(v.encode('utf-8')).decode('utf-8')
        catalog = load_catalog(data['identifier'][0], **properties)
        return catalog.load_table((data['identifier'][1], data['identifier'][2]))
    raise TypeError(f'do not know how to deserialize {classname}')