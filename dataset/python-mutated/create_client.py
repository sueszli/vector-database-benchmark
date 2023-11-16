"""A sample for creating a Storage Transfer Service client."""
from google.cloud import storage_transfer

def create_transfer_client():
    if False:
        for i in range(10):
            print('nop')
    return storage_transfer.StorageTransferServiceClient()