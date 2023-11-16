import sys
sys.path.append('..')
from configure.settings import DBSelector
from configure.util import mongo_convert_df
db_name = 'db_stock'
doc_name = 'fund_component_159941'

def get_df():
    if False:
        for i in range(10):
            print('nop')
    client = DBSelector().mongo('qq')
    doc = client[db_name][doc_name]
    df = mongo_convert_df(doc)
    return df

def weight(df):
    if False:
        for i in range(10):
            print('nop')
    df['weight'] = df['weight'].map(lambda x: float(x.replace('%', '')))
    date_df = df.set_index(['chn_name', 'date']).unstack()['weight'].sort_index()
    date_df = date_df.fillna(0)
    date_df.to_excel('nsda1.xlsx', encoding='utf8')

def scale(df):
    if False:
        for i in range(10):
            print('nop')
    pass

def main():
    if False:
        return 10
    df = get_df()
    scale(df)
if __name__ == '__main__':
    main()