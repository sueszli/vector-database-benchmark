from ..BaseRecommender import BaseRecommender

class ImplicitBaseRecommender(BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        if False:
            print('Hello World!')
        super(ImplicitBaseRecommender, self).__init__(URM_train=URM_train)
        self.verbose = verbose

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_not_compute=None, remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        if False:
            return 10
        list_tuples_item_score = self.rec.recommend(user_id_array, self.URM_train, filter_already_liked_items=remove_seen_flag, N=cutoff, filter_items=items_to_not_compute)
        if return_scores:
            return list_tuples_item_score
        else:
            list_items = []
            for tuple in list_tuples_item_score:
                item = tuple[0]
                list_items.append(item)
            return list_items