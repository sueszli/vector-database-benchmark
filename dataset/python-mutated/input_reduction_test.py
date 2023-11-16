from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import InputReduction

class TestInputReduction(AllenNlpTestCase):

    def test_input_reduction(self):
        if False:
            i = 10
            return i + 15
        inputs = {'sentence': 'I always write unit tests for my code.'}
        archive = load_archive(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive)
        reducer = InputReduction(predictor)
        reduced = reducer.attack_from_json(inputs, 'tokens', 'grad_input_1')
        assert reduced is not None
        assert 'final' in reduced
        assert 'original' in reduced
        assert reduced['final'][0]
        assert len(reduced['final'][0]) <= len(reduced['original'])
        for word in reduced['final'][0]:
            assert word in reduced['original']
        inputs = {'sentence': 'Eric Wallace was an intern at AI2'}
        archive = load_archive(self.FIXTURES_ROOT / 'simple_tagger' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'sentence_tagger')
        reducer = InputReduction(predictor)
        reduced = reducer.attack_from_json(inputs, 'tokens', 'grad_input_1')
        assert reduced is not None
        assert 'final' in reduced
        assert 'original' in reduced
        for reduced_input in reduced['final']:
            assert reduced_input
            assert len(reduced_input) <= len(reduced['original'])
            for word in reduced_input:
                assert word in reduced['original']