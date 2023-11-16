from __future__ import absolute_import, division, print_function
from .ciderD_scorer import CiderScorer

class CiderD:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, n=4, sigma=6.0, df='corpus'):
        if False:
            for i in range(10):
                print('nop')
        self._n = n
        self._sigma = sigma
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        if False:
            return 10
        '\n        Main function to compute CIDEr score\n        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>\n                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>\n        :return: cider (float) : computed CIDEr score for the corpus\n        '
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            tmp_cider_scorer += (hypo[0], ref)
        (score, scores) = tmp_cider_scorer.compute_score()
        return (score, scores)

    def method(self):
        if False:
            print('Hello World!')
        return 'CIDEr-D'