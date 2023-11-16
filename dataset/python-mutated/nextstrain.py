import logging
import os
import subprocess
import urllib
import torch
from .util import _mkdir_p, get_data_directory
DATA = get_data_directory(__file__)
URL = 'https://github.com/pyro-ppl/datasets/raw/master/nextstrain.data.pt.gz'

def load_nextstrain_counts(map_location=None) -> dict:
    if False:
        return 10
    "\n    Loads a SARS-CoV-2 dataset.\n\n    The original dataset is a preprocessed intermediate ``metadata.tsv.gz`` available via\n    `nextstrain <https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html>`_.\n    The ``metadata.tsv.gz`` file was then aggregated to\n    (month,location,lineage) and (lineage,mutation) bins by the Broad Institute's\n    `preprocessing script <https://github.com/broadinstitute/pyro-cov/blob/master/scripts/preprocess_nextstrain.py>`_.\n    "
    _mkdir_p(DATA)
    basename = URL.split('/')[-1]
    gz_filename = os.path.join(DATA, basename)
    if not os.path.exists(gz_filename):
        logging.debug(f'downloading {URL}')
        urllib.request.urlretrieve(URL, gz_filename)
    filename = gz_filename.replace('.gz', '')
    if not os.path.exists(filename):
        logging.debug(f'unzipping {gz_filename}')
        subprocess.check_call(['gunzip', '-k', gz_filename])
    if map_location is None:
        map_location = torch.tensor(0.0).device
    return torch.load(filename, map_location=map_location)