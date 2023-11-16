"""
Test BLEUScore metric against reference
"""
from neon.transforms.cost import BLEUScore

def test_bleuscore():
    if False:
        return 10
    sentences = ['a quick brown fox jumped', 'the rain in spain falls mainly on the plains']
    references = [['a fast brown fox jumped', 'a quick brown fox vaulted', 'a rapid fox of brown color jumped', 'the dog is running on the grass'], ['the precipitation in spain falls on the plains', 'spanish rain falls for the most part on the plains', 'the rain in spain falls in the plains most of the time', 'it is raining today']]
    bleu_score_references = [92.9, 88.0, 81.5, 67.1]
    bleu_metric = BLEUScore()
    bleu_metric(sentences, references)
    for (score, reference) in zip(bleu_metric.bleu_n, bleu_score_references):
        assert round(score, 1) == reference
if __name__ == '__main__':
    test_bleuscore()