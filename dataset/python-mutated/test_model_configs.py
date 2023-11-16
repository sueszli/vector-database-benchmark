import ludwig.datasets

def test_default_model_config(tmpdir):
    if False:
        i = 10
        return i + 15
    titanic_configs = ludwig.datasets.model_configs_for_dataset('titanic')
    assert len(titanic_configs) > 0
    titanic = ludwig.datasets.get_dataset('titanic', cache_dir=tmpdir)
    assert titanic.default_model_config is not None
    assert titanic.default_model_config == titanic_configs['default']

def test_best_model_config(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    higgs_configs = ludwig.datasets.model_configs_for_dataset('higgs')
    assert len(higgs_configs) > 0
    higgs = ludwig.datasets.get_dataset('higgs', cache_dir=tmpdir)
    assert higgs.default_model_config is not None
    assert higgs.best_model_config is not None
    assert higgs.default_model_config == higgs_configs['default']
    assert higgs.best_model_config == higgs_configs['best']

def test_dataset_has_no_model_configs(tmpdir):
    if False:
        print('Hello World!')
    bbc_news_configs = ludwig.datasets.model_configs_for_dataset('bbcnews')
    assert len(bbc_news_configs) == 0
    bbcnews = ludwig.datasets.get_dataset('bbcnews', cache_dir=tmpdir)
    assert bbcnews.default_model_config is None