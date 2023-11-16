from recommenders.datasets import movielens

def test_movielens_privacy():
    if False:
        for i in range(10):
            print('nop')
    'Check that there are no privacy concerns. In Movielens, we check that all the\n    userID are numbers.\n    '
    df = movielens.load_pandas_df(size='100k')
    users = df['userID'].values.tolist()
    assert all((isinstance(x, int) for x in users))