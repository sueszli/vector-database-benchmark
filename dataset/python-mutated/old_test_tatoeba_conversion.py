import os
import tempfile
import unittest
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import DEFAULT_REPO, TatoebaConverter
from transformers.testing_utils import slow
from transformers.utils import cached_property

@unittest.skipUnless(os.path.exists(DEFAULT_REPO), 'Tatoeba directory does not exist.')
class TatoebaConversionTester(unittest.TestCase):

    @cached_property
    def resolver(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = tempfile.mkdtemp()
        return TatoebaConverter(save_dir=tmp_dir)

    @slow
    def test_resolver(self):
        if False:
            for i in range(10):
                print('nop')
        self.resolver.convert_models(['heb-eng'])

    @slow
    def test_model_card(self):
        if False:
            for i in range(10):
                print('nop')
        (content, mmeta) = self.resolver.write_model_card('opus-mt-he-en', dry_run=True)
        assert mmeta['long_pair'] == 'heb-eng'