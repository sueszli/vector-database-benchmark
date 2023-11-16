"""
Created on 28/11/2020

@author: Alessandro Sanvito
"""
from ..Base.BaseRecommender import BaseRecommender

class MergedHybridRecommender(BaseRecommender):
    """
    This recommender merges two recommendes by weighting their ratings
    """
    RECOMMENDER_NAME = 'MergedHybridRecommender'

    def __init__(self, URM_train, recommender1: BaseRecommender, recommender2: BaseRecommender, verbose=True):
        if False:
            return 10
        self.RECOMMENDER_NAME = recommender1.RECOMMENDER_NAME[:-11] + recommender2.RECOMMENDER_NAME[:-11] + 'HybridRecommender'
        super(MergedHybridRecommender, self).__init__(URM_train, verbose=verbose)
        self.recommender1 = recommender1
        self.recommender2 = recommender2

    def fit(self, alpha=0.5):
        if False:
            while True:
                i = 10
        self.alpha = alpha

    def save_model(self, folder_path, file_name=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if False:
            for i in range(10):
                print('nop')
        return self.alpha * self.recommender1._compute_item_score(user_id_array, items_to_compute) + (1 - self.alpha) * self.recommender2._compute_item_score(user_id_array, items_to_compute)