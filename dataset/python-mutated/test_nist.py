"""
Tests for NIST translation evaluation metric
"""
import io
import unittest
from nltk.data import find
from nltk.translate.nist_score import corpus_nist

class TestNIST(unittest.TestCase):

    def test_sentence_nist(self):
        if False:
            return 10
        ref_file = find('models/wmt15_eval/ref.ru')
        hyp_file = find('models/wmt15_eval/google.ru')
        mteval_output_file = find('models/wmt15_eval/mteval-13a.output')
        with open(mteval_output_file) as mteval_fin:
            mteval_nist_scores = map(float, mteval_fin.readlines()[-4].split()[1:-1])
        with open(ref_file, encoding='utf8') as ref_fin:
            with open(hyp_file, encoding='utf8') as hyp_fin:
                hypotheses = list(map(lambda x: x.split(), hyp_fin))
                references = list(map(lambda x: [x.split()], ref_fin))
                for (i, mteval_nist) in zip(range(1, 10), mteval_nist_scores):
                    nltk_nist = corpus_nist(references, hypotheses, i)
                    assert abs(mteval_nist - nltk_nist) < 0.05