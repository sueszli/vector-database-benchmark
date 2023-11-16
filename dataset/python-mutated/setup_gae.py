"""Python execution environment setup for scripts that require GAE."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tarfile
from typing import Optional, Sequence
from . import common
_PARSER = argparse.ArgumentParser(description='\nPython execution environment setup for scripts that require GAE.\n')
GAE_DOWNLOAD_ZIP_PATH = os.path.join('.', 'gae-download.zip')

def main(args: Optional[Sequence[str]]=None) -> None:
    if False:
        return 10
    'Runs the script to setup GAE.'
    unused_parsed_args = _PARSER.parse_args(args=args)
    sys.path.append('.')
    sys.path.append(common.GOOGLE_APP_ENGINE_SDK_HOME)
    for (directory, _, files) in os.walk('.'):
        for file_name in files:
            if file_name.endswith('.pyc'):
                filepath = os.path.join(directory, file_name)
                os.remove(filepath)
    print('Checking whether google-cloud-sdk is installed in %s' % common.GOOGLE_CLOUD_SDK_HOME)
    if not os.path.exists(common.GOOGLE_CLOUD_SDK_HOME):
        print('Downloading Google Cloud SDK (this may take a little while)...')
        os.makedirs(common.GOOGLE_CLOUD_SDK_HOME)
        try:
            common.url_retrieve('https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-364.0.0-linux-x86_64.tar.gz', 'gcloud-sdk.tar.gz')
        except Exception as e:
            print('Error downloading Google Cloud SDK. Exiting.')
            raise Exception('Error downloading Google Cloud SDK.') from e
        print('Download complete. Installing Google Cloud SDK...')
        tar = tarfile.open(name='gcloud-sdk.tar.gz')
        tar.extractall(path=os.path.join(common.OPPIA_TOOLS_DIR, 'google-cloud-sdk-364.0.0/'))
        tar.close()
        os.remove('gcloud-sdk.tar.gz')
    subprocess.run([common.GCLOUD_PATH, 'components', 'install', 'beta', 'cloud-datastore-emulator', 'app-engine-python', 'app-engine-python-extras', '--quiet'], check=True)
if __name__ == '__main__':
    main()