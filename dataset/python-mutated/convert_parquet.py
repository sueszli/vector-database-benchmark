import json
import pandas as pd

def reformat_csv_to_openassistant(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Reformat the downloaded CSV into either Instruction or Text format\n    so that it could be directly ingested into the training pipeline.\n\n    Parameters\n    ----------\n    df: the downloaded panda dataframe\n\n    Return\n    ------\n    DataFrame: reformatted dataframe\n    '
    new_df = pd.DataFrame()
    new_df['INSTRUCTION'] = df['question_title']
    new_df['RESPONSE'] = df['content']
    new_df['SOURCE'] = 'Zhihu'
    new_df['METADATA'] = df.apply(lambda x: json.dumps({'question_id': x['question_id'], 'answer_id': x['answer_id'], 'author_id': x['author_id'], 'upvotes': x['upvotes'], 'answer_creation_time': x['answer_creation_time']}, ensure_ascii=False), axis=1)
    new_df = new_df[~(new_df['RESPONSE'] == ' ') | new_df['RESPONSE'].isna()]
    return new_df
if __name__ == '__main__':
    input_csv = 'zhihu.csv'
    df = pd.read_csv(input_csv)
    df = reformat_csv_to_openassistant(df)
    df.to_parquet('dataset.parquet', row_group_size=100, engine='pyarrow', index=False)