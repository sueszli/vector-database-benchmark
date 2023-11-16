"""
Created on 2015-06-22
Update  on 2017-05-16
Author: Lockvictor/片刻
《推荐系统实践》协同过滤算法源代码
参考地址: https://github.com/Lockvictor/MovieLens-RecSys
更新地址: https://github.com/apachecn/AiLearning
"""
from __future__ import print_function
import math
import random
import sys
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
random.seed(0)

class ItemBasedCF:
    """ TopN recommendation - ItemBasedCF """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.train_mat = {}
        self.test_mat = {}
        self.n_users = 0
        self.n_items = 0
        self.n_sim_item = 20
        self.n_rec_item = 10
        self.item_mat_similarity = {}
        self.item_popular = {}
        self.item_count = 0
        print('Similar item number = %d' % self.n_sim_item, file=sys.stderr)
        print('Recommended item number = %d' % self.n_rec_item, file=sys.stderr)

    def splitData(self, dataFile, test_size):
        if False:
            i = 10
            return i + 15
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(dataFile, sep='\t', names=header)
        self.n_users = df.user_id.unique().shape[0]
        self.n_items = df.item_id.unique().shape[0]
        print('Number of users = ' + str(self.n_users) + ' | Number of items = ' + str(self.n_items))
        (self.train_data, self.test_data) = cv.train_test_split(df, test_size=test_size)
        print('分离训练集和测试集成功', file=sys.stderr)
        print('len(train) = %s' % np.shape(self.train_data)[0], file=sys.stderr)
        print('len(test) = %s' % np.shape(self.test_data)[0], file=sys.stderr)

    def calc_similarity(self):
        if False:
            return 10
        self.train_mat = np.zeros((self.n_users, self.n_items))
        for line in self.train_data.itertuples():
            self.train_mat[int(line.user_id) - 1, int(line.item_id) - 1] = float(line.rating)
        self.test_mat = np.zeros((self.n_users, self.n_items))
        for line in self.test_data.itertuples():
            self.test_mat[int(line.user_id) - 1, int(line.item_id) - 1] = float(line.rating)
        print('1:', np.shape(np.mat(self.train_mat).T))
        self.item_mat_similarity = pairwise_distances(np.mat(self.train_mat).T, metric='cosine')
        print('item_mat_similarity=', np.shape(self.item_mat_similarity), file=sys.stderr)
        print('开始统计流行item的数量...', file=sys.stderr)
        for i_index in range(self.n_items):
            if np.sum(self.train_mat[:, i_index]) != 0:
                self.item_popular[i_index] = np.sum(self.train_mat[:, i_index] != 0)
        self.item_count = len(self.item_popular)
        print('总共流行item数量 = %d' % self.item_count, file=sys.stderr)

    def recommend(self, u_index):
        if False:
            print('Hello World!')
        'recommend(找出top K的电影，对电影进行相似度sum的排序，取出top N的电影数)\n\n        Args:\n            u_index   用户_ID-1=用户index\n        Returns:\n            rec_item  电影推荐列表，按照相似度从大到小的排序\n        '
        ' Find K similar items and recommend N items. '
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        i_items = np.where(self.train_mat[u_index, :] != 0)[0]
        watched_items = dict(zip(i_items, self.train_mat[u_index, i_items]))
        for (i_item, rating) in watched_items.items():
            i_other_items = np.where(self.item_mat_similarity[i_item, :] != 0)[0]
            for (related_item, w) in sorted(dict(zip(i_other_items, self.item_mat_similarity[i_item, i_other_items])).items(), key=itemgetter(1), reverse=True)[0:K]:
                if related_item in watched_items:
                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += w * rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        ' return precision, recall, coverage and popularity '
        print('Evaluation start...', file=sys.stderr)
        hit = 0
        rec_count = 0
        test_count = 0
        all_rec_items = set()
        popular_sum = 0
        for u_index in range(50):
            if u_index > 0 and u_index % 10 == 0:
                print('recommended for %d users' % u_index, file=sys.stderr)
            print('u_index', u_index)
            rec_items = self.recommend(u_index)
            print('rec_items=', rec_items)
            for (item, _) in rec_items:
                if self.test_mat[u_index, item] != 0:
                    hit += 1
                    print('self.test_mat[%d, %d]=%s' % (u_index, item, self.test_mat[u_index, item]))
                if item in self.item_popular:
                    popular_sum += math.log(1 + self.item_popular[item])
            rec_count += len(rec_items)
            test_count += np.sum(self.test_mat[u_index, :] != 0)
        print('-------', hit, rec_count)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        popularity = popular_sum / (1.0 * rec_count)
        print('precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (precision, recall, coverage, popularity), file=sys.stderr)
if __name__ == '__main__':
    dataFile = 'data/16.RecommenderSystems/ml-100k/u.data'
    itemcf = ItemBasedCF()
    itemcf.splitData(dataFile, test_size=0.3)
    itemcf.calc_similarity()
    print('推荐结果', itemcf.recommend(u_index=1))
    print('---', np.where(itemcf.test_mat[1, :] != 0)[0])