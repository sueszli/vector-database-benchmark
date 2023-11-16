from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender


class ModelMergingRecommender(ItemKNNCustomSimilarityRecommender):

    RECOMMENDER_NAME = "ModelMergingRecommender"

    def fit(self, rec1, rec2, alpha=0.5, selectTopK=False, topK=100):

        sim_merged = (1 - alpha) * rec1.W_sparse + alpha * rec2.W_sparse

        super(ModelMergingRecommender, self).fit(sim_merged, selectTopK, topK)
