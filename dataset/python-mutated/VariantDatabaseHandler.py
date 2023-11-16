from UM.Settings.SQLQueryFactory import SQLQueryFactory
from UM.Settings.DatabaseContainerMetadataController import DatabaseMetadataContainerController
from UM.Settings.InstanceContainer import InstanceContainer

class VariantDatabaseHandler(DatabaseMetadataContainerController):
    """The Database handler for Variant containers"""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(SQLQueryFactory(table='variant', fields={'id': 'text', 'name': 'text', 'hardware_type': 'text', 'definition': 'text', 'version': 'text', 'setting_version': 'text'}))
        self._container_type = InstanceContainer