"""
Helper methods to handle files in remote locations.
"""
import logging
import os
from pathlib import Path
import requests
from samcli.lib.utils.progressbar import progressbar
from samcli.local.lambdafn.zip import unzip
LOG = logging.getLogger(__name__)

def unzip_from_uri(uri, layer_zip_path, unzip_output_dir, progressbar_label):
    if False:
        return 10
    '\n    Download the LayerVersion Zip to the Layer Pkg Cache\n\n    Parameters\n    ----------\n    uri str\n        Uri to download from\n    layer_zip_path str\n        Path to where the content from the uri should be downloaded to\n    unzip_output_dir str\n        Path to unzip the zip to\n    progressbar_label str\n        Label to use in the Progressbar\n    '
    try:
        get_request = requests.get(uri, stream=True, verify=os.environ.get('AWS_CA_BUNDLE', True))
        with open(layer_zip_path, 'wb') as local_layer_file:
            file_length = int(get_request.headers['Content-length'])
            with progressbar(file_length, progressbar_label) as p_bar:
                for data in get_request.iter_content(chunk_size=None):
                    local_layer_file.write(data)
                    p_bar.update(len(data))
        unzip(layer_zip_path, unzip_output_dir, permission=448)
    finally:
        path_to_layer = Path(layer_zip_path)
        if path_to_layer.exists():
            path_to_layer.unlink()