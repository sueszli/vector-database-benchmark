import os
import pytest
import ray
import time

def main():
    if False:
        while True:
            i = 10
    'Submits custom resource request.\n\n    Also, validates runtime env data submitted with the Ray Job that executes\n    this script.\n    '
    assert pytest.__version__ == '6.0.0'
    assert os.getenv('key_foo') == 'value_bar'
    ray.autoscaler.sdk.request_resources(bundles=[{'Custom2': 3}, {'Custom2': 3}, {'Custom2': 3}])
    while ray.cluster_resources().get('Custom2', 0) < 3 and ray.cluster_resources().get('Custom2', 0) < 6:
        time.sleep(0.1)
    print('Submitted custom scale request!')
if __name__ == '__main__':
    ray.init('auto')
    main()