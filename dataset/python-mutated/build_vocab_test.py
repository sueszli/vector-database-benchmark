import os
import sys
import pytest
from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary

class TestBuildVocabCommand(AllenNlpTestCase):

    def test_build_vocab(self):
        if False:
            for i in range(10):
                print('nop')
        output_path = self.TEST_DIR / 'vocab.tar.gz'
        sys.argv = ['allennlp', 'build-vocab', str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet'), str(output_path)]
        main()
        assert os.path.exists(output_path)
        vocab = Vocabulary.from_files(output_path)
        vocab.get_token_index('neg', 'labels') == 0
        with pytest.raises(RuntimeError, match='already exists'):
            main()
        sys.argv.append('--force')
        main()