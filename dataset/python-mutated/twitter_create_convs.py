import json
from pathlib import Path
import polars as pl
from tqdm import tqdm
path_string = 'PUT THE PATH HERE TO WHERE YOU STORED THE PARQUET FILES'
folder_path = Path(path_string)
processed_folder_path = folder_path / 'processed'
output_path = folder_path / 'twitter-conv-trees.jsonl'
parq_files = sorted(processed_folder_path.rglob('*.parquet'))
wanted_cols = ['timestamp_ms', 'id', 'text', 'truncated', 'in_reply_to_status_id', 'in_reply_to_user_id', 'is_quote_status', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'filter_level', 'lang', 'possibly_sensitive', 'hashtags', 'user_id', 'user_verified', 'user_followers_count', 'user_statuses_count']
df_list = []
for p in parq_files:
    df_list.append(pl.read_parquet(p, columns=wanted_cols))
p_df = pl.concat(df_list)
del df_list
p_df_replies_only = p_df.filter(pl.col('in_reply_to_status_id').is_null().is_not())
p_df_group_reply_to_status = p_df_replies_only.groupby('in_reply_to_status_id').count().sort('count', reverse=True)
group_reply_parq = folder_path / 'group_reply_parq.parquet'
p_df_group_reply_to_status.write_parquet(group_reply_parq)
p_join = p_df.join(p_df_group_reply_to_status, left_on='id', right_on='in_reply_to_status_id', how='inner')
tweets_that_have_replies_path = folder_path / 'tweets_that_have_replies.parquet'
p_join.write_parquet(tweets_that_have_replies_path)
tweets_that_are_replies_path = folder_path / 'tweets_that_are_replies.parquet'
p_df_replies_only.write_parquet(tweets_that_are_replies_path)
origin_tweets = p_join.filter(pl.col('in_reply_to_status_id').is_null() & (pl.col('lang') == 'en'))

def role_decide(user_id, prompt_user):
    if False:
        return 10
    if user_id == prompt_user:
        return 'prompter'
    else:
        return 'assistant'

class ConversationTreeNode:

    def __init__(self, tweet_id, prompt_user, from_df, children_df, metadata=None):
        if False:
            i = 10
            return i + 15
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = from_df.filter(pl.col('id') == tweet_id).to_dicts()[0]
        self.metadata['prompt_user'] = prompt_user
        self.role = role_decide(self.metadata['user_id'], prompt_user)
        self.children = None
        self.text = self.metadata['text']
        del self.metadata['text']
        self.get_children(tweet_id=tweet_id, children_df=children_df)

    def get_children(self, tweet_id, children_df):
        if False:
            return 10
        children_dicts = children_df.filter(pl.col('in_reply_to_status_id') == tweet_id).to_dicts()
        if len(children_dicts) > 0:
            children = [ConversationTreeNode(tweet_id=c['id'], prompt_user=self.metadata['prompt_user'], from_df=children_df, children_df=children_df, metadata=c) for c in children_dicts]
            self.children = children

class ConversationTree:

    def __init__(self, tweet_id, prompt_user, from_df, children_df, r_metadata=None):
        if False:
            return 10
        self.root = ConversationTreeNode(tweet_id=tweet_id, prompt_user=prompt_user, from_df=from_df, children_df=children_df, metadata=r_metadata)
        self.metadata = None
conv_tree_list = [ConversationTree(tweet_id=r['id'], prompt_user=r['user_id'], from_df=origin_tweets, children_df=p_df_replies_only, r_metadata=r) for r in tqdm(origin_tweets.to_dicts())]
with open(output_path, 'w') as output:
    for t in tqdm(conv_tree_list):
        json.dump(obj=t, fp=output, default=lambda x: x.__dict__)
        output.write('\n')