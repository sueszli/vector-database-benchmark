import json
from pathlib import Path
import huggingface_hub
import pytest
import requests
test_dir = Path(__file__).parent
with open(test_dir.parent / 'src/serge/data/models.json', 'r') as models_file:
    families = json.load(models_file)
checks = []
for family in families:
    for model in family['models']:
        for file in model['files']:
            checks.append((model['repo'], file['filename']))

@pytest.mark.parametrize('repo,filename', checks)
def test_model_available(repo, filename):
    if False:
        while True:
            i = 10
    url = huggingface_hub.hf_hub_url(repo, filename, repo_type='model', revision='main')
    r = requests.head(url)
    assert r.ok, f'Model {repo}/{filename} not available'