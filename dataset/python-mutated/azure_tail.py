from io import BytesIO
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from metaflow.exception import MetaflowException
from metaflow.plugins.azure.blob_service_client_factory import get_azure_blob_service_client
from metaflow.plugins.azure.azure_utils import parse_azure_full_path

class AzureTail(object):

    def __init__(self, blob_full_uri):
        if False:
            while True:
                i = 10
        'Location should be something like <container_name>/blob'
        (container_name, blob_name) = parse_azure_full_path(blob_full_uri)
        if not blob_name:
            raise MetaflowException(msg='Failed to parse blob_full_uri into <container_name>/<blob_name> (got %s)' % blob_full_uri)
        service = get_azure_blob_service_client()
        container = service.get_container_client(container_name)
        self._blob_client = container.get_blob_client(blob_name)
        self._pos = 0
        self._tail = b''

    def __iter__(self):
        if False:
            print('Hello World!')
        buf = self._fill_buf()
        if buf is not None:
            for line in buf:
                if line.endswith(b'\n'):
                    yield line
                else:
                    self._tail = line
                    break

    def _make_range_request(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._blob_client.download_blob(offset=self._pos).readall()
        except ResourceNotFoundError:
            return None
        except HttpResponseError as e:
            if e.status_code != 416:
                print('Failed to tail log from step (status code = %d)' % (e.status_code,))
            return None
        except Exception as e:
            print('Failed to tail log from step (%s)' % type(e))
            return None

    def _fill_buf(self):
        if False:
            return 10
        data = self._make_range_request()
        if data is None:
            return None
        if data:
            buf = BytesIO(data)
            self._pos += len(data)
            self._tail = b''
            return buf
        else:
            return None
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tail an Azure Blob. Must specify METAFLOW_AZURE_STORAGE_BLOB_SERVICE_ENDPOINT in environment.')
    parser.add_argument('blob_full_uri', help='The blob to tail. Format is <container_name>/<blob>')
    args = parser.parse_args()
    az_tail = AzureTail(args.blob_full_uri)
    for line in az_tail:
        print(line.strip().decode('utf-8'))