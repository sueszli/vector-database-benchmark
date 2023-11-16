from __future__ import annotations
from typing import Any, Dict
from sentry import options
from sentry.db.models import control_silo_only_model
from sentry.models.files.abstractfileblob import AbstractFileBlob
from sentry.models.files.control_fileblobowner import ControlFileBlobOwner
from sentry.options.manager import UnknownOption
from sentry.tasks.files import delete_file_control

def control_file_storage_config() -> Dict[str, Any] | None:
    if False:
        for i in range(10):
            print('nop')
    "\n    When sentry is deployed in a siloed mode file relations\n    used by control silo models are stored separately from\n    region silo resources.\n\n    While we consistently write to the ControlFile django\n    model for control silo resources, we can't ensure\n    that each deployment has separate control + region storage\n    backends. We coalesce those options here. None means use the\n    global default storage options.\n    "
    try:
        storage_backend = options.get('filestore.control.backend')
        storage_options = options.get('filestore.control.options')
        if storage_backend:
            return {'backend': storage_backend, 'options': storage_options}
    except UnknownOption:
        pass
    return None

@control_silo_only_model
class ControlFileBlob(AbstractFileBlob):

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_controlfileblob'
    FILE_BLOB_OWNER_MODEL = ControlFileBlobOwner
    DELETE_FILE_TASK = delete_file_control

    @classmethod
    def _storage_config(cls) -> Dict[str, Any] | None:
        if False:
            while True:
                i = 10
        return control_file_storage_config()