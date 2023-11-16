from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps

class ScoresHybridRecommender_4(BaseRecommender):

    RECOMMENDER_NAME = "ScoresHybridRecommender_4"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, recommender_4):
        super(ScoresHybridRecommender_4, self).__init__(URM_train)

        self.alpha = None
        self.beta = None
        self.gamma = None

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4

    def fit(self, alpha=0.5, beta=0.5, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array)

        item_weights = self.gamma * (self.alpha * (self.beta * item_weights_1 + (1 - self.beta) * item_weights_2) + (1 - self.alpha) * item_weights_3) + (1 - self.gamma) * item_weights_4

        return item_weights

    def save_model(self, folder_path, file_name=None):
        return

    def load_model(self, folder_path, file_name=None):
        return
