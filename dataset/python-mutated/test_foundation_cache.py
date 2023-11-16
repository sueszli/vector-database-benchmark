import glob
import os
import shutil
import tempfile
import pytest
import stanza
from stanza.models.common.foundation_cache import FoundationCache, load_charlm
from stanza.tests import TEST_MODELS_DIR
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_charlm_cache():
    if False:
        return 10
    models_path = os.path.join(TEST_MODELS_DIR, 'en', 'backward_charlm', '*')
    models = glob.glob(models_path)
    assert len(models) >= 1
    model_file = models[0]
    cache = FoundationCache()
    with tempfile.TemporaryDirectory(dir='.') as test_dir:
        temp_file = os.path.join(test_dir, 'charlm.pt')
        shutil.copy2(model_file, temp_file)
        model = load_charlm(temp_file)
        model = cache.load_charlm(temp_file)
    with pytest.raises(FileNotFoundError):
        model = load_charlm(temp_file)
    model = cache.load_charlm(temp_file)