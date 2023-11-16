from __future__ import annotations
import warnings
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.google.cloud.transfers.azure_blob_to_gcs import AzureBlobStorageToGCSOperator as AzureBlobStorageToGCSOperatorFromGoogleProvider

class AzureBlobStorageToGCSOperator(AzureBlobStorageToGCSOperatorFromGoogleProvider):
    """
    This class is deprecated.

    Please use
    :class:`airflow.providers.google.cloud.transfers.azure_blob_to_gcs.AzureBlobStorageToGCSOperator`.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        warnings.warn('This class is deprecated.\n            Please use\n            `airflow.providers.google.cloud.transfers.azure_blob_to_gcs.AzureBlobStorageToGCSOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)