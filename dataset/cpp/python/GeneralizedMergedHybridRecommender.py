# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2020

@author: Alessandro Sanvito
"""

from Recommenders.BaseRecommender import BaseRecommender


class GeneralizedMergedHybridRecommender(BaseRecommender):
    """
    This recommender merges two recommenders by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            recommenders: list,
            verbose=True
    ):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedMergedHybridRecommender, self).__init__(
            URM_train,
            verbose=verbose
        )

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas

    def save_model(self, folder_path, file_name=None):
        pass

    # Ok so basically in first line it computes the item scores for the first recommender multiplied its alpha
    # and from second line it adds all the other recommender scores multiplied by their alphas
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.recommenders[0]._compute_item_score(user_id_array, items_to_compute)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.recommenders[index]._compute_item_score(user_id_array,
                                                                                                items_to_compute)
        return result
