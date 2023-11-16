"""A sample for creating a Storage Transfer Service client."""
import googleapiclient.discovery

def create_transfer_client():
    if False:
        for i in range(10):
            print('nop')
    return googleapiclient.discovery.build('storagetransfer', 'v1')