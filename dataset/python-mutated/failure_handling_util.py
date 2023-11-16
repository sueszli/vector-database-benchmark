"""Util of GCE specifics to ingegrate with WorkerPreemptionHandler."""
import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
GCP_METADATA_HEADER = {'Metadata-Flavor': 'Google'}
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_RESTARTABLE_EXIT_CODE = 143
GRACE_PERIOD_GCE = 3600

def gce_exit_fn():
    if False:
        print('Hello World!')
    sys.exit(_RESTARTABLE_EXIT_CODE)

def default_tpu_exit_fn():
    if False:
        print('Hello World!')
    'Default exit function to run after saving checkpoint for TPUStrategy.\n\n  For TPUStrategy, we want the coordinator to exit after workers are down so\n  that restarted coordinator would not connect to workers scheduled to be\n  preempted. This function achieves so by attempting to get a key-value store\n  from coordination service, which will block until workers are done and then\n  returns with error. Then we have the coordinator sys.exit(42) to re-schedule\n  the job.\n  '
    logging.info('Waiting for workers to exit...')
    try:
        context.context().get_config_key_value('BLOCK_TILL_EXIT')
    except:
        logging.info('Restarting cluster due to preemption.')
        sys.exit(42)

def request_compute_metadata(path):
    if False:
        i = 10
        return i + 15
    'Returns GCE VM compute metadata.'
    gce_metadata_endpoint = 'http://' + os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
    req = request.Request('%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, path), headers={'Metadata-Flavor': 'Google'})
    info = request.urlopen(req).read()
    if isinstance(info, bytes):
        return info.decode('utf-8')
    else:
        return info

def termination_watcher_function_gce():
    if False:
        for i in range(10):
            print('nop')
    result = request_compute_metadata('instance/maintenance-event') == 'TERMINATE_ON_HOST_MAINTENANCE'
    return result

def on_gcp():
    if False:
        for i in range(10):
            print('nop')
    'Detect whether the current running environment is on GCP.'
    gce_metadata_endpoint = 'http://' + os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
    try:
        response = requests.get('%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, 'instance/hostname'), headers=GCP_METADATA_HEADER, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@enum.unique
class PlatformDevice(enum.Enum):
    INTERNAL_CPU = 'internal_CPU'
    INTERNAL_GPU = 'internal_GPU'
    INTERNAL_TPU = 'internal_TPU'
    GCE_GPU = 'GCE_GPU'
    GCE_TPU = 'GCE_TPU'
    GCE_CPU = 'GCE_CPU'
    UNSUPPORTED = 'unsupported'

def detect_platform():
    if False:
        for i in range(10):
            print('nop')
    'Returns the platform and device information.'
    if on_gcp():
        if context.context().list_logical_devices('GPU'):
            return PlatformDevice.GCE_GPU
        elif context.context().list_logical_devices('TPU'):
            return PlatformDevice.GCE_TPU
        else:
            return PlatformDevice.GCE_CPU
    elif context.context().list_logical_devices('GPU'):
        return PlatformDevice.INTERNAL_GPU
    elif context.context().list_logical_devices('TPU'):
        return PlatformDevice.INTERNAL_TPU
    else:
        return PlatformDevice.INTERNAL_CPU