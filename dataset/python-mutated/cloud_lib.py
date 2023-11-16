"""Utilities that interact with cloud service.
"""
import requests
GCP_METADATA_URL = 'http://metadata/computeMetadata/v1/instance/hostname'
GCP_METADATA_HEADER = {'Metadata-Flavor': 'Google'}

def on_gcp():
    if False:
        i = 10
        return i + 15
    'Detect whether the current running environment is on GCP.'
    try:
        response = requests.get(GCP_METADATA_URL, headers=GCP_METADATA_HEADER, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False