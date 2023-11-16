import os
import sys
import numpy as np
import pandas as pd
import config.set_content as setting
from middleware.utils import pd_load, pd_like, pd_save, pd_rename, get_days

def data_converting(infile, outfile):
    if False:
        print('Hello World!')
    '\n    # 将用户交易数据转化为: \n    # 将\n    #     用户ID、各种基金、变动金额、时间\n    # 转化为：\n    #     用户ID、基金ID、购买金额、时间的数据\n    '
    print('Loading user daliy data...')
    df = pd_load(infile)
    df['money'] = df['变动金额'].apply(lambda line: abs(line))
    df_user_item = df.groupby(['用户账号', '证券代码'], as_index=False).agg({'money': np.sum}).sort_values('money', ascending=True)
    pd_rename(df_user_item, ['user_id', 'item_id', 'rating'])
    pd_save(df_user_item, outfile)

def create_user2item(infile, outfile):
    if False:
        while True:
            i = 10
    '创建user-item评分矩阵'
    print('Loading user daliy data...')
    df_user_item = pd_load(infile)
    user_id = sorted(df_user_item['user_id'].unique(), reverse=False)
    item_id = sorted(df_user_item['item_id'].unique(), reverse=False)
    rating_matrix = np.zeros([len(user_id), len(item_id)])
    rating_matrix = pd.DataFrame(rating_matrix, index=user_id, columns=item_id)
    print('Converting data...')
    count = 0
    user_num = len(user_id)
    for uid in user_id:
        user_rating = df_user_item[df_user_item['user_id'] == uid].drop(['user_id'], axis=1)
        user_rated_num = len(user_rating)
        for row in range(0, user_rated_num):
            item_id = user_rating['item_id'].iloc[row]
            rating_matrix.loc[uid, item_id] = user_rating['rating'].iloc[row]
        count += 1
        if count % 10 == 0:
            completed_percentage = round(float(count) / user_num * 100)
            print('Completed %s' % completed_percentage + '%')
    rating_matrix.index.name = 'user_id'
    pd_save(rating_matrix, outfile, index=True)

def create_item2feature(infile, outfile):
    if False:
        for i in range(10):
            print('nop')
    '创建 item-特征-是否存在 矩阵'
    print('Loading item feature data...')
    df_item_info = pd_load(infile, header=1)
    items_num = df_item_info.shape[0]
    columns = df_item_info.columns.tolist()
    new_cols = [col for col in columns if col not in ['info_type', 'info_investype']]
    info_types = sorted(df_item_info['info_type'].unique(), reverse=False)
    info_investypes = sorted(df_item_info['info_investype'].unique(), reverse=False)
    dict_n_cols = {'info_type': info_types, 'info_investype': info_investypes}
    new_cols.append(dict_n_cols)

    def get_new_columns(new_cols):
        if False:
            while True:
                i = 10
        new_columns = []
        for col in new_cols:
            if isinstance(col, dict):
                for (k, vs) in col.items():
                    new_columns += vs
            else:
                new_columns.append(col)
        return new_columns
    new_columns = get_new_columns(new_cols)

    def deal_line(line, new_cols):
        if False:
            while True:
                i = 10
        result = []
        for col in new_cols:
            if isinstance(col, str):
                result.append(line[col])
            elif isinstance(col, dict):
                for (k, vs) in col.items():
                    for v in vs:
                        if v == line[k]:
                            result.append(1)
                        else:
                            result.append(0)
            else:
                print('类型错误')
                sys.exit(1)
        return result
    df = df_item_info.apply(lambda line: deal_line(line, new_cols), axis=1, result_type='expand')
    pd_rename(df, new_columns)
    end_time = '2020-10-19'
    df['days'] = df['info_creattime'].apply(lambda str_time: get_days(str_time, end_time))
    df.drop(['info_name', 'info_foundlevel', 'info_creattime'], axis=1, inplace=True)
    pd_save(df, outfile)

def rs_1_data_preprocess():
    if False:
        for i in range(10):
            print('nop')
    data_infile = setting.PATH_CONFIG['user_daily']
    user_infile = setting.PATH_CONFIG['user_item']
    user_outfile = setting.PATH_CONFIG['matrix_user_item2rating']
    item_infile = setting.PATH_CONFIG['item_info']
    item_outfile = setting.PATH_CONFIG['matrix_item2feature']
    if not os.path.exists(user_infile):
        '数据处理部分'
        data_converting(data_infile, user_infile)
        create_user2item(user_infile, user_outfile)
    elif not os.path.exists(user_outfile):
        create_user2item(user_infile, user_outfile)
    if not os.path.exists(item_outfile):
        create_item2feature(item_infile, item_outfile)
    user_feature = pd_load(user_outfile)
    item_feature = pd_load(item_outfile)
    user_feature.set_index('user_id', inplace=True)
    item_feature.set_index('item_id', inplace=True)
    return (user_feature, item_feature)

def cos_measure(item_feature_vector, user_rated_items_matrix):
    if False:
        print('Hello World!')
    '\n    计算item之间的余弦夹角相似度\n    :param item_feature_vector: 待测量的item特征向量\n    :param user_rated_items_matrix: 用户已评分的items的特征矩阵\n    :return: 待计算item与用户已评分的items的余弦夹角相识度的向量\n    '
    x_c = item_feature_vector * user_rated_items_matrix.T + 1e-07
    mod_x = np.sqrt(item_feature_vector * item_feature_vector.T)
    mod_c = np.sqrt((user_rated_items_matrix * user_rated_items_matrix.T).diagonal())
    cos_xc = x_c / (mod_x * mod_c)
    return cos_xc

def comp_user_feature(user_rated_vector, item_feature_matrix):
    if False:
        print('Hello World!')
    '\n    根据user的评分来计算得到user的喜好特征\n    :param user_rated_vector  : user的评分向量\n    :param item_feature_matrix: item的特征矩阵\n    :return: user的喜好特征\n    '
    user_rating_mean = user_rated_vector.mean()
    user_like_item = user_rated_vector.loc[user_rated_vector >= user_rating_mean]
    user_unlike_item = user_rated_vector.loc[user_rated_vector < user_rating_mean]
    print('user_like_item: \n', user_like_item)
    print('user_unlike_item: \n', user_unlike_item)
    user_like_item_index = map(int, user_like_item.index.values)
    user_unlike_item_index = map(int, user_unlike_item.index.values)
    user_like_item_rating = np.matrix(user_like_item.values)
    user_unlike_item_rating = np.matrix(user_unlike_item.values)
    user_like_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_like_item_index, :].values)
    user_unlike_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unlike_item_index, :].values)
    weight_of_like = user_like_item_rating / user_like_item_rating.sum()
    weight_of_unlike = user_unlike_item_rating / user_unlike_item_rating.sum()
    print('weight_of_like: ', weight_of_like)
    print('weight_of_unlike: ', weight_of_unlike)
    user_like_feature = weight_of_like * user_like_item_feature_matrix
    user_unlike_feature = weight_of_unlike * user_unlike_item_feature_matrix
    user_feature_tol = user_like_feature - user_unlike_feature
    return user_feature_tol

def rs_2_cb_recommend(user_feature, item_feature_matrix, K=20):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算得到与user最相似的Top K个item推荐给user\n    :param user_feature: 待推荐用户的对item的评分向量\n    :param item_feature_matrix: 包含所有item的特征矩阵\n    :param K: 推荐给user的item数量\n    :return: 与user最相似的Top K个item的编号\n    '
    user_rated_vector = user_feature.loc[user_feature > 0]
    user_unrated_vector = user_feature
    user_item_feature_tol = comp_user_feature(user_rated_vector, item_feature_matrix)
    print('>>> 用户调性', user_item_feature_tol)
    user_unrated_item_index = map(int, user_unrated_vector.index.values)
    user_unrated_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unrated_item_index, :].values)
    similarity = list(np.array(cos_measure(user_item_feature_tol, user_unrated_item_feature_matrix))[0])
    key = {'item_index': list(user_unrated_vector.index.values), 'similarity': similarity}
    item_sim_df = pd.DataFrame(key)
    item_sim_df.sort_values('similarity', ascending=False, inplace=True)
    return item_sim_df.iloc[:K, 0].values

def estimate_rate(user_rated_vector, similarity):
    if False:
        return 10
    '\n    估计用户对item的评分\n    :param user_rated_vector: 用户已有item评分向量\n    :param similarity: 待估计item和已评分item的相识度向量\n    :return:用户对item的评分的估计\n    '
    rate_hat = user_rated_vector * similarity.T / similarity.sum()
    return rate_hat[0, 0]

def rs_2_cb_recommend_estimate(user_feature, item_feature_matrix, item):
    if False:
        i = 10
        return i + 15
    '\n    基于内容的推荐算法对item的评分进行估计\n    :param item_feature_matrix: 包含所有item的特征矩阵\n    :param user_feature: 待估计用户的对item的评分向量\n    :param item: 待估计item的编号\n    :return: 基于内容的推荐算法对item的评分进行估计\n    '
    user_item_index = user_feature.index
    user_rated_vector = np.matrix(user_feature.loc[user_feature > 0].values)
    user_rated_items = map(int, user_item_index[user_feature > 0].values)
    user_rated_items_matrix = np.matrix(item_feature_matrix.loc[user_rated_items, :].values)
    item_feature_vector = np.matrix(item_feature_matrix.loc[item].values)
    cos_xc = cos_measure(item_feature_vector, user_rated_items_matrix)
    rate_hat = estimate_rate(user_rated_vector, cos_xc)
    return rate_hat

def main():
    if False:
        print('Hello World!')
    user_id = 20200930
    K = 10
    (user_feature, item_feature) = rs_1_data_preprocess()
    user_feature = user_feature.loc[user_id, :]
    print('>>> 1 \n', user_feature)
    result = rs_2_cb_recommend(user_feature, item_feature, K)
    print(result)
if __name__ == '__main__':
    main()