import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from tests.integration_tests.utils import private_test

@private_test
def test_get_config_and_load(tmpdir):
    if False:
        while True:
            i = 10
    yosemite_config = ludwig.datasets._get_dataset_config('yosemite')
    assert isinstance(yosemite_config, DatasetConfig)
    yosemite_dataset = ludwig.datasets.get_dataset('yosemite', cache_dir=tmpdir)
    assert isinstance(yosemite_dataset, DatasetLoader)
    df = yosemite_dataset.load()
    assert df is not None
    assert len(df) == 18721

def test_get_config_kaggle(tmpdir):
    if False:
        i = 10
        return i + 15
    twitter_bots_config = ludwig.datasets._get_dataset_config('twitter_bots')
    assert isinstance(twitter_bots_config, DatasetConfig)
    twitter_bots_dataset = ludwig.datasets.get_dataset('twitter_bots', cache_dir=tmpdir)
    assert isinstance(twitter_bots_dataset, DatasetLoader)
    assert twitter_bots_dataset.is_kaggle_dataset