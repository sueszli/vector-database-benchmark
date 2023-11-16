import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
expected_from_rte_feature_extration = '\nalwayson        => True\nne_hyp_extra    => 0\nne_overlap      => 1\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 3\nword_overlap    => 3\n\nalwayson        => True\nne_hyp_extra    => 0\nne_overlap      => 1\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 2\nword_overlap    => 1\n\nalwayson        => True\nne_hyp_extra    => 1\nne_overlap      => 1\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 1\nword_overlap    => 2\n\nalwayson        => True\nne_hyp_extra    => 1\nne_overlap      => 0\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 6\nword_overlap    => 2\n\nalwayson        => True\nne_hyp_extra    => 1\nne_overlap      => 0\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 4\nword_overlap    => 0\n\nalwayson        => True\nne_hyp_extra    => 1\nne_overlap      => 0\nneg_hyp         => 0\nneg_txt         => 0\nword_hyp_extra  => 3\nword_overlap    => 1\n'

class TestRTEClassifier:

    def test_rte_feature_extraction(self):
        if False:
            for i in range(10):
                print('nop')
        pairs = rte_corpus.pairs(['rte1_dev.xml'])[:6]
        test_output = [f'{key:<15} => {rte_features(pair)[key]}' for pair in pairs for key in sorted(rte_features(pair))]
        expected_output = expected_from_rte_feature_extration.strip().split('\n')
        expected_output = list(filter(None, expected_output))
        assert test_output == expected_output

    def test_feature_extractor_object(self):
        if False:
            while True:
                i = 10
        rtepair = rte_corpus.pairs(['rte3_dev.xml'])[33]
        extractor = RTEFeatureExtractor(rtepair)
        assert extractor.hyp_words == {'member', 'China', 'SCO.'}
        assert extractor.overlap('word') == set()
        assert extractor.overlap('ne') == {'China'}
        assert extractor.hyp_extra('word') == {'member'}

    def test_rte_classification_without_megam(self):
        if False:
            print('Hello World!')
        clf = rte_classifier('IIS', sample_N=100)
        clf = rte_classifier('GIS', sample_N=100)

    def test_rte_classification_with_megam(self):
        if False:
            return 10
        try:
            config_megam()
        except (LookupError, AttributeError) as e:
            pytest.skip('Skipping tests with dependencies on MEGAM')
        clf = rte_classifier('megam', sample_N=100)