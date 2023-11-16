import os
import pytest
try:
    from recommenders.models.deeprec.deeprec_utils import prepare_hparams, download_deeprec_resources, load_yaml
except ImportError:
    pass

@pytest.mark.parametrize('must_exist_attributes', ['FEATURE_COUNT', 'data_format', 'dim'])
@pytest.mark.gpu
def test_prepare_hparams(deeprec_resource_path, must_exist_attributes):
    if False:
        while True:
            i = 10
    data_path = os.path.join(deeprec_resource_path, 'xdeepfm')
    yaml_file = os.path.join(data_path, 'xDeepFM.yaml')
    if not os.path.exists(yaml_file):
        download_deeprec_resources('https://recodatasets.z20.web.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')
    hparams = prepare_hparams(yaml_file)
    assert hasattr(hparams, must_exist_attributes)

@pytest.mark.gpu
def test_load_yaml_file(deeprec_resource_path):
    if False:
        i = 10
        return i + 15
    data_path = os.path.join(deeprec_resource_path, 'xdeepfm')
    yaml_file = os.path.join(data_path, 'xDeepFM.yaml')
    if not os.path.exists(yaml_file):
        download_deeprec_resources('https://recodatasets.z20.web.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')
    config = load_yaml(yaml_file)
    assert config is not None