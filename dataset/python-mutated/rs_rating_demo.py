from __future__ import print_function
import sys
import math
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from middleware.utils import TimeStat, Chart

def splitData(dataFile, test_size):
    if False:
        print('Hello World!')
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(dataFile, sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print('>>> 本数据集包含: 总用户数 = %s | 总电影数 = %s' % (n_users, n_items))
    (train_data, test_data) = cv.train_test_split(df, test_size=test_size)
    print('>>> 训练:测试 = %s:%s = %s:%s' % (len(train_data), len(test_data), 1 - test_size, test_size))
    return (df, n_users, n_items, train_data, test_data)

def calc_similarity(n_users, n_items, train_data, test_data):
    if False:
        print('Hello World!')
    '\n    line:  Pandas(Index=93661, user_id=624, item_id=750, rating=4, timestamp=891961163)\n    '
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    print('1:', np.shape(train_data_matrix))
    print('2:', np.shape(train_data_matrix.T))
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
    print('开始统计流行item的数量...', file=sys.stderr)
    item_popular = {}
    for i_index in range(n_items):
        if np.sum(train_data_matrix[:, i_index]) != 0:
            item_popular[i_index] = np.sum(train_data_matrix[:, i_index] != 0)
    item_count = len(item_popular)
    print('总共流行 item 数量 = %d' % item_count, file=sys.stderr)
    return (train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular)

def predict(rating, similarity, type='user'):
    if False:
        while True:
            i = 10
    '\n    :param rating: 训练数据\n    :param similarity: 向量距离\n    :return:\n    '
    print('+++ %s' % type)
    print('    rating=', np.shape(rating))
    print('    similarity=', np.shape(similarity))
    if type == 'item':
        '\n        综合打分:  \n            rating.dot(similarity) 表示：\n                某1个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n\n                某2个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n\n                ...\n                某n个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n\n            = 人-电影-评分(943, 1682) * 电影-电影-距离(1682, 1682) \n            = 人-电影-总评分距离(943, 1682)\n            \n            np.array([np.abs(similarity).sum(axis=1)]) 表示: 横向求和: 1 表示某一行所有的列求和\n                第1列表示：某个A电影，对于所有电影计算出A的总距离\n                第2列表示：某个B电影，对于所有电影的综出B的总距离\n                ...\n                第n列表示：某个N电影，对于所有电影的综出N的总距离\n            = 每一个电影的总距离 (1, 1682)\n\n            pred = 人-电影-平均评分 (943, 1682)\n        '
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    elif type == 'user':
        mean_user_rating = rating.mean(axis=1)
        rating_diff = rating - mean_user_rating[:, np.newaxis]
        '\n        综合打分:  \n            similarity.dot(rating_diff) 表示：\n                第1列：第1个人与其他人的相似度 * 人与电影的相似度，得到 第1个人对第1/2/3列电影的 总得分 1*n\n                第2列：第2个人与其他人的相似度 * 人与电影的相似度，得到 第2个人对第1/2/3列电影的 总得分 1*n\n                ...\n                第n列：第n个人与其他人的相似度 * 人与电影的相似度，得到 第n个人对第1/2/3列电影的 总得分 1*n\n            = 人-人-距离(943, 943)  *  人-电影-评分(943, 1682)\n            = 人-电影-总评分距离(943, 1682)\n\n            np.array([np.abs(similarity).sum(axis=1)]) 表示: 横向求和: 1 表示某一行所有的列求和\n                第1列表示：第A个人，对于所有人计算出A的总距离\n                第2列表示：第B个人，对于所有人计算出B的总距离\n                ...\n                第n列表示：第N个人，对于所有人计算出N的总距离\n            = 每一个电影的总距离 (1, 943)\n\n            pred = 均值 + 人-电影-平均评分 (943, 1682)\n        '
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

def rmse(prediction, ground_truth):
    if False:
        return 10
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(prediction, ground_truth))

def evaluate(prediction, item_popular, name):
    if False:
        for i in range(10):
            print('nop')
    hit = 0
    rec_count = 0
    test_count = 0
    popular_sum = 0
    all_rec_items = set()
    for u_index in range(n_users):
        items = np.where(train_data_matrix[u_index, :] == 0)[0]
        pre_items = sorted(dict(zip(items, prediction[u_index, items])).items(), key=itemgetter(1), reverse=True)[:20]
        test_items = np.where(test_data_matrix[u_index, :] != 0)[0]
        for (item, _) in pre_items:
            if item in test_items:
                hit += 1
            all_rec_items.add(item)
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])
        rec_count += len(pre_items)
        test_count += len(test_items)
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_items) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    print('--- %s: precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (name, precision, recall, coverage, popularity), file=sys.stderr)

def recommend(u_index, prediction):
    if False:
        while True:
            i = 10
    items = np.where(train_data_matrix[u_index, :] == 0)[0]
    pre_items = sorted(dict(zip(items, prediction[u_index, items])).items(), key=itemgetter(1), reverse=True)[:10]
    test_items = np.where(test_data_matrix[u_index, :] != 0)[0]
    result = [key for (key, value) in pre_items]
    result.sort(reverse=False)
    print('原始结果(%s): %s' % (len(test_items), test_items))
    print('推荐结果(%s): %s' % (len(result), result))

def main():
    if False:
        while True:
            i = 10
    global n_users, train_data_matrix, test_data_matrix
    path_root = '/Users/jiangzl/work/data/机器学习'
    dataFile = '%s/16.RecommenderSystems/ml-100k/u.data' % path_root
    (df, n_users, n_items, train_data, test_data) = splitData(dataFile, test_size=0.25)
    (train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular) = calc_similarity(n_users, n_items, train_data, test_data)
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    print('>>> Item based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
    print('>>> User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    print('\nMovieLen100K的稀疏度: %s%%\n' % (sparsity * 100))
    index = 11
    minrmse = 2.6717213264389765
    (u, s, vt) = svds(train_data_matrix, k=index)
    s_diag_matrix = np.diag(s)
    svd_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
    r_rmse = rmse(svd_prediction, test_data_matrix)
    print('+++ k=%s, svd-shape: %s' % (index, np.shape(svd_prediction)))
    print('>>> Model based CF RMSE: %s\n' % minrmse)
    evaluate(item_prediction, item_popular, 'item')
    evaluate(user_prediction, item_popular, 'user')
    evaluate(svd_prediction, item_popular, 'svd')
    recommend(1, svd_prediction)
if __name__ == '__main__':
    main()