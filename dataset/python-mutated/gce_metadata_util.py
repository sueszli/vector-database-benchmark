"""Fetches GCE metadata if the calling process is running on a GCE VM."""
import requests
BASE_METADATA_URL = 'http://metadata/computeMetadata/v1/'

def _fetch_metadata(key):
    if False:
        for i in range(10):
            print('nop')
    try:
        headers = {'Metadata-Flavor': 'Google'}
        uri = BASE_METADATA_URL + key
        resp = requests.get(uri, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.text
    except requests.exceptions.RequestException:
        pass
    return ''

def _fetch_custom_gce_metadata(customMetadataKey):
    if False:
        for i in range(10):
            print('nop')
    return _fetch_metadata('instance/attributes/' + customMetadataKey)

def fetch_dataflow_job_id():
    if False:
        return 10
    return _fetch_custom_gce_metadata('job_id')