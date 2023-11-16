import bigdl.orca.data.pandas
from bigdl.orca.data.transformer import *
path = 'answer_correctness/train.csv'
used_data_types_list = ['timestamp', 'user_id', 'content_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
data_shard = bigdl.orca.data.pandas.read_csv(path, usecols=used_data_types_list, index_col=0)

def get_feature(df):
    if False:
        i = 10
        return i + 15
    feature_df = df.iloc[:int(9 / 10 * len(df))]
    return feature_df
feature_shard = data_shard.transform_shard(get_feature)

def get_train_questions_only(df):
    if False:
        return 10
    train_questions_only_df = df[df['answered_correctly'] != -1]
    return train_questions_only_df
train_questions_only_shard = feature_shard.transform_shard(get_train_questions_only)
train_questions_only_shard = train_questions_only_shard.group_by(columns='user_id', agg={'answered_correctly': ['mean', 'count', 'stddev', 'skewness']}, join=True)
target = 'answered_correctly'

def filter_non_target(df):
    if False:
        for i in range(10):
            print('nop')
    train_df = df[df[target] != -1]
    return train_df
train_shard = train_questions_only_shard.transform_shard(filter_non_target)

def fill_na(df, val):
    if False:
        i = 10
        return i + 15
    train_df = df.fillna(value=val)
    return train_df
train_shard = train_shard.transform_shard(fill_na, 0.5)