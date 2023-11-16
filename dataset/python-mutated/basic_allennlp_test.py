import pytest
from allennlp.common.testing import AllenNlpTestCase

@pytest.mark.skip('makes test-install fail (and also takes 30 seconds)')
class TestBasicAllenNlp(AllenNlpTestCase):

    @classmethod
    def test_run_as_script(cls):
        if False:
            i = 10
            return i + 15
        import tutorials.tagger.basic_allennlp