import numpy as np
from numpy.core.getlimits import inf
import random
import math
class itemCF:
    def __init__(self, dl, sm):
        self.img2idx = dl.item2index
        self.idx2img = dl.items_list
        self.img2lab = dl.filelist
        self.sim_mat = sm.collaborative_similarity_matrix
    
    def getRatingRank(self, rating_dict):
        ratingRank = sorted(list(rating_dict.items()), key=lambda x:x[1], reverse=True)
        return ratingRank
        
    def recommend(self, top, rating_dict, shuffle=False):
        ratingRank = self.getRatingRank(rating_dict)
        rec_item_num_dict = {}
        total_rate = sum([i[1] for i in ratingRank])
        max_rate = 0
        min_rate = inf
        rec_rank_with_rate = []
        rec_rank = []
        total_idx_pool = []
        for item in ratingRank:
            rec_item_num_dict[item[0]] = int(item[1] / total_rate * top) + 1
            
        for item in ratingRank:
            img_name = item[0]
            img_rate = item[1]
            max_rate = max(img_rate, max_rate)
            min_rate = min(img_rate, min_rate)
            idx = self.img2idx[img_name]
            topn = rec_item_num_dict[img_name]
            idx_rank = np.argsort(-self.sim_mat[idx])
            this_idx_rank = []
            counter = 0
            for idx in idx_rank:
                if counter >= topn:
                    break
                if idx in total_idx_pool:
                    continue
                total_idx_pool.append(idx)
                this_idx_rank.append(idx)
                counter += 1
                
            img_rank = [(self.idx2img[i], img_rate) for i in this_idx_rank]
            for img_rate in img_rank:
                for i in img_rank:
                    if i[0] == img_rate[0]:
                        continue
                rec_rank_with_rate.append(img_rate)
        
        if shuffle:
            last_idx = 0
            score = max_rate
            while score >= min_rate:
                bucket = []
                for idx in range(last_idx, len(rec_rank_with_rate)):
                    img, rate = rec_rank_with_rate[idx]
                    if rate == score:
                        bucket.append(img)
                    elif rate != score:
                        last_idx = idx + 1
                random.shuffle(bucket)
                for img in bucket:
                    rec_rank.append(img)
                score -= 1
        else:
            rec_rank = [i[0] for i in rec_rank_with_rate]
        return rec_rank[:top]