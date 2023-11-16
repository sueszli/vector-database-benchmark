from UM.Settings.SQLQueryFactory import SQLQueryFactory
from UM.Settings.DatabaseContainerMetadataController import DatabaseMetadataContainerController
from UM.Settings.InstanceContainer import InstanceContainer

class IntentDatabaseHandler(DatabaseMetadataContainerController):
    """The Database handler for Intent containers"""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(SQLQueryFactory(table='intent', fields={'id': 'text', 'name': 'text', 'quality_type': 'text', 'intent_category': 'text', 'variant': 'text', 'definition': 'text', 'material': 'text', 'version': 'text', 'setting_version': 'text', 'icon': 'text', 'description': 'text'}))
        self._container_type = InstanceContainer