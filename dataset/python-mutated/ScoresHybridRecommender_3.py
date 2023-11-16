from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps

class ScoresHybridRecommender_3(BaseRecommender):
    RECOMMENDER_NAME = 'ScoresHybridRecommender_3'

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        if False:
            while True:
                i = 10
        super(ScoresHybridRecommender_3, self).__init__(URM_train)
        self.alpha = None
        self.beta = None
        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3

    def fit(self, alpha=0.5, beta=0.5):
        if False:
            print('Hello World!')
        self.alpha = alpha
        self.beta = beta

    def _compute_item_score(self, user_id_array, items_to_compute):
        if False:
            i = 10
            return i + 15
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        item_weights = item_weights_1 * self.alpha * self.beta + item_weights_2 * self.alpha * (1 - self.beta) + item_weights_3 * (1 - self.alpha)
        return item_weights

    def save_model(self, folder_path, file_name=None):
        if False:
            return 10
        return

    def load_model(self, folder_path, file_name=None):
        if False:
            for i in range(10):
                print('nop')
        return