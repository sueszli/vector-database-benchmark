"""Provides app identity services."""
from __future__ import annotations
from core import feconf
_GCS_RESOURCE_BUCKET_NAME_SUFFIX = '-resources'

def get_application_id() -> str:
    if False:
        i = 10
        return i + 15
    "Returns the application's App Engine ID.\n\n    Locally we set the GOOGLE_CLOUD_PROJECT environment variable in\n    scripts/servers.py when starting the dev server. In production\n    the GOOGLE_CLOUD_PROJECT is set by the server.\n\n    Returns:\n        str. The application ID.\n\n    Raises:\n        ValueError. Value can't be None for application id.\n    "
    app_id = feconf.OPPIA_PROJECT_ID
    if app_id is None:
        raise ValueError('Value None for application id is invalid.')
    return app_id

def get_gcs_resource_bucket_name() -> str:
    if False:
        for i in range(10):
            print('nop')
    "Returns the application's bucket name for GCS resources, which depends\n    on the application ID in production mode, or default bucket name in\n    development mode.\n\n    This needs to be in sync with deploy.py which adds the bucket name to\n    constants.ts\n\n    Also, note that app_identity.get_default_gcs_bucket_name() returns None\n    if we try to use it in production mode but the default bucket hasn't been\n    enabled through the project console.\n\n    Returns:\n        str. The bucket name for the application's GCS resources.\n    "
    return get_application_id() + _GCS_RESOURCE_BUCKET_NAME_SUFFIX