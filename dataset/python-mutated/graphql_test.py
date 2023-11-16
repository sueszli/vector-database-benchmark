from common import *
if vaex.utils.devmode:
    pytest.skip('runs too slow when developing', allow_module_level=True)

@pytest.fixture(scope='module')
def schema(ds_trimmed_cache):
    if False:
        for i in range(10):
            print('nop')
    ds_trimmed_cache = ds_trimmed_cache.drop('123456')
    return ds_trimmed_cache.graphql.schema()

@pytest.fixture()
def df(df_trimmed):
    if False:
        return 10
    return df_trimmed.drop('123456')

def test_aggregates(df, schema):
    if False:
        return 10
    result = schema.execute('\n    {\n        df {\n            count\n            min {\n                x\n                y\n            }\n            mean {\n                x\n                y\n            }\n            max {\n                x\n                y\n            }\n        }\n    }\n    ')
    assert not result.errors
    assert result.data['df']['count'] == len(df)
    assert result.data['df']['min']['x'] == df.x.min()
    assert result.data['df']['min']['y'] == df.y.min()
    assert result.data['df']['max']['x'] == df.x.max()
    assert result.data['df']['max']['y'] == df.y.max()
    assert result.data['df']['mean']['x'] == df.x.mean()
    assert result.data['df']['mean']['y'] == df.y.mean()

def test_groupby(df, schema):
    if False:
        for i in range(10):
            print('nop')
    result = schema.execute('\n    {\n        df {\n            groupby {\n                x {\n                    min {\n                        x\n                    }\n                }\n            }\n        }\n    }\n    ')
    assert not result.errors
    dfg = df.groupby('x', agg={'xmin': vaex.agg.min('x')})
    assert result.data['df']['groupby']['x']['min']['x'] == dfg['xmin'].tolist()

def test_row_pagination(df, schema):
    if False:
        for i in range(10):
            print('nop')

    def values(row, name):
        if False:
            while True:
                i = 10
        return [k[name] for k in row]
    result = schema.execute('\n    {\n        df {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df.x.tolist()
    result = schema.execute('\n    {\n        df {\n            row(offset: 2) { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[2:].x.tolist()
    result = schema.execute('\n    {\n        df {\n            row(limit: 2) { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[:2].x.tolist()
    result = schema.execute('\n    {\n        df {\n            row(offset: 3, limit: 2) { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[3:5].x.tolist()

def test_where(df, schema):
    if False:
        return 10

    def values(row, name):
        if False:
            print('Hello World!')
        return [k[name] for k in row]
    result = schema.execute('\n    {\n        df(where: {x: {_eq: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x == 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {x: {_neq: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x != 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {x: {_gt: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x > 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {x: {_gte: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x >= 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {x: {_lt: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x < 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {x: {_lte: 4}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x <= 4].x.tolist()
    result = schema.execute('\n    {\n        df(where: {_not: {x: {_lte: 4}}}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[~(df.x <= 4)].x.tolist()
    result = schema.execute('\n    {\n        df(where: {_or: [{x: {_eq: 4}}, {x: {_eq: 6}} ]}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == [4, 6]
    result = schema.execute('\n    {\n        df(where: {_and: [{x: {_gte: 4}}, {x: {_lte: 6}} ]}) {\n            row { x }\n        }\n    }\n    ')
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == [4, 5, 6]

def test_pandas(df, schema):
    if False:
        while True:
            i = 10
    df_pandas = df.to_pandas_df()

    def values(row, name):
        if False:
            while True:
                i = 10
        return [k[name] for k in row]
    result = df_pandas.graphql.execute('\n    {\n        df(where: {x: {_eq: 4}}) {\n            row { x }\n        }\n    }\n    ')