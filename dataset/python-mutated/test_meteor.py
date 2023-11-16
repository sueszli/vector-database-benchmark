import unittest
from nltk.translate.meteor_score import meteor_score

class TestMETEOR(unittest.TestCase):
    reference = [['this', 'is', 'a', 'test'], ['this', 'istest']]
    candidate = ['THIS', 'Is', 'a', 'tEST']

    def test_meteor(self):
        if False:
            print('Hello World!')
        score = meteor_score(self.reference, self.candidate, preprocess=str.lower)
        assert score == 0.9921875

    def test_reference_type_check(self):
        if False:
            for i in range(10):
                print('nop')
        str_reference = [' '.join(ref) for ref in self.reference]
        self.assertRaises(TypeError, meteor_score, str_reference, self.candidate)

    def test_candidate_type_check(self):
        if False:
            i = 10
            return i + 15
        str_candidate = ' '.join(self.candidate)
        self.assertRaises(TypeError, meteor_score, self.reference, str_candidate)