from pytest import raises
from elasticsearch_dsl import aggs, query

def test_repr():
    if False:
        return 10
    max_score = aggs.Max(field='score')
    a = aggs.A('terms', field='tags', aggs={'max_score': max_score})
    assert "Terms(aggs={'max_score': Max(field='score')}, field='tags')" == repr(a)

def test_meta():
    if False:
        return 10
    max_score = aggs.Max(field='score')
    a = aggs.A('terms', field='tags', aggs={'max_score': max_score}, meta={'some': 'metadata'})
    assert {'terms': {'field': 'tags'}, 'aggs': {'max_score': {'max': {'field': 'score'}}}, 'meta': {'some': 'metadata'}} == a.to_dict()

def test_meta_from_dict():
    if False:
        for i in range(10):
            print('nop')
    max_score = aggs.Max(field='score')
    a = aggs.A('terms', field='tags', aggs={'max_score': max_score}, meta={'some': 'metadata'})
    assert aggs.A(a.to_dict()) == a

def test_A_creates_proper_agg():
    if False:
        i = 10
        return i + 15
    a = aggs.A('terms', field='tags')
    assert isinstance(a, aggs.Terms)
    assert a._params == {'field': 'tags'}

def test_A_handles_nested_aggs_properly():
    if False:
        for i in range(10):
            print('nop')
    max_score = aggs.Max(field='score')
    a = aggs.A('terms', field='tags', aggs={'max_score': max_score})
    assert isinstance(a, aggs.Terms)
    assert a._params == {'field': 'tags', 'aggs': {'max_score': max_score}}

def test_A_passes_aggs_through():
    if False:
        return 10
    a = aggs.A('terms', field='tags')
    assert aggs.A(a) is a

def test_A_from_dict():
    if False:
        print('Hello World!')
    d = {'terms': {'field': 'tags'}, 'aggs': {'per_author': {'terms': {'field': 'author.raw'}}}}
    a = aggs.A(d)
    assert isinstance(a, aggs.Terms)
    assert a._params == {'field': 'tags', 'aggs': {'per_author': aggs.A('terms', field='author.raw')}}
    assert a['per_author'] == aggs.A('terms', field='author.raw')
    assert a.aggs.per_author == aggs.A('terms', field='author.raw')

def test_A_fails_with_incorrect_dict():
    if False:
        return 10
    correct_d = {'terms': {'field': 'tags'}, 'aggs': {'per_author': {'terms': {'field': 'author.raw'}}}}
    with raises(Exception):
        aggs.A(correct_d, field='f')
    d = correct_d.copy()
    del d['terms']
    with raises(Exception):
        aggs.A(d)
    d = correct_d.copy()
    d['xx'] = {}
    with raises(Exception):
        aggs.A(d)

def test_A_fails_with_agg_and_params():
    if False:
        while True:
            i = 10
    a = aggs.A('terms', field='tags')
    with raises(Exception):
        aggs.A(a, field='score')

def test_buckets_are_nestable():
    if False:
        i = 10
        return i + 15
    a = aggs.Terms(field='tags')
    b = a.bucket('per_author', 'terms', field='author.raw')
    assert isinstance(b, aggs.Terms)
    assert b._params == {'field': 'author.raw'}
    assert a.aggs == {'per_author': b}

def test_metric_inside_buckets():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.Terms(field='tags')
    b = a.metric('max_score', 'max', field='score')
    assert a is b
    assert a.aggs['max_score'] == aggs.Max(field='score')

def test_buckets_equals_counts_subaggs():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.Terms(field='tags')
    a.bucket('per_author', 'terms', field='author.raw')
    b = aggs.Terms(field='tags')
    assert a != b

def test_buckets_to_dict():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.Terms(field='tags')
    a.bucket('per_author', 'terms', field='author.raw')
    assert {'terms': {'field': 'tags'}, 'aggs': {'per_author': {'terms': {'field': 'author.raw'}}}} == a.to_dict()
    a = aggs.Terms(field='tags')
    a.metric('max_score', 'max', field='score')
    assert {'terms': {'field': 'tags'}, 'aggs': {'max_score': {'max': {'field': 'score'}}}} == a.to_dict()

def test_nested_buckets_are_reachable_as_getitem():
    if False:
        while True:
            i = 10
    a = aggs.Terms(field='tags')
    b = a.bucket('per_author', 'terms', field='author.raw')
    assert a['per_author'] is not b
    assert a['per_author'] == b

def test_nested_buckets_are_settable_as_getitem():
    if False:
        while True:
            i = 10
    a = aggs.Terms(field='tags')
    b = a['per_author'] = aggs.A('terms', field='author.raw')
    assert a.aggs['per_author'] is b

def test_filter_can_be_instantiated_using_positional_args():
    if False:
        i = 10
        return i + 15
    a = aggs.Filter(query.Q('term', f=42))
    assert {'filter': {'term': {'f': 42}}} == a.to_dict()
    assert a == aggs.A('filter', query.Q('term', f=42))

def test_filter_aggregation_as_nested_agg():
    if False:
        print('Hello World!')
    a = aggs.Terms(field='tags')
    a.bucket('filtered', 'filter', query.Q('term', f=42))
    assert {'terms': {'field': 'tags'}, 'aggs': {'filtered': {'filter': {'term': {'f': 42}}}}} == a.to_dict()

def test_filter_aggregation_with_nested_aggs():
    if False:
        print('Hello World!')
    a = aggs.Filter(query.Q('term', f=42))
    a.bucket('testing', 'terms', field='tags')
    assert {'filter': {'term': {'f': 42}}, 'aggs': {'testing': {'terms': {'field': 'tags'}}}} == a.to_dict()

def test_filters_correctly_identifies_the_hash():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.A('filters', filters={'group_a': {'term': {'group': 'a'}}, 'group_b': {'term': {'group': 'b'}}})
    assert {'filters': {'filters': {'group_a': {'term': {'group': 'a'}}, 'group_b': {'term': {'group': 'b'}}}}} == a.to_dict()
    assert a.filters.group_a == query.Q('term', group='a')

def test_bucket_sort_agg():
    if False:
        i = 10
        return i + 15
    bucket_sort_agg = aggs.BucketSort(sort=[{'total_sales': {'order': 'desc'}}], size=3)
    assert bucket_sort_agg.to_dict() == {'bucket_sort': {'sort': [{'total_sales': {'order': 'desc'}}], 'size': 3}}
    a = aggs.DateHistogram(field='date', interval='month')
    a.bucket('total_sales', 'sum', field='price')
    a.bucket('sales_bucket_sort', 'bucket_sort', sort=[{'total_sales': {'order': 'desc'}}], size=3)
    assert {'date_histogram': {'field': 'date', 'interval': 'month'}, 'aggs': {'total_sales': {'sum': {'field': 'price'}}, 'sales_bucket_sort': {'bucket_sort': {'sort': [{'total_sales': {'order': 'desc'}}], 'size': 3}}}} == a.to_dict()

def test_bucket_sort_agg_only_trnunc():
    if False:
        while True:
            i = 10
    bucket_sort_agg = aggs.BucketSort(**{'from': 1, 'size': 1})
    assert bucket_sort_agg.to_dict() == {'bucket_sort': {'from': 1, 'size': 1}}
    a = aggs.DateHistogram(field='date', interval='month')
    a.bucket('bucket_truncate', 'bucket_sort', **{'from': 1, 'size': 1})
    assert {'date_histogram': {'field': 'date', 'interval': 'month'}, 'aggs': {'bucket_truncate': {'bucket_sort': {'from': 1, 'size': 1}}}} == a.to_dict()

def test_geohash_grid_aggregation():
    if False:
        return 10
    a = aggs.GeohashGrid(**{'field': 'centroid', 'precision': 3})
    assert {'geohash_grid': {'field': 'centroid', 'precision': 3}} == a.to_dict()

def test_geotile_grid_aggregation():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.GeotileGrid(**{'field': 'centroid', 'precision': 3})
    assert {'geotile_grid': {'field': 'centroid', 'precision': 3}} == a.to_dict()

def test_boxplot_aggregation():
    if False:
        while True:
            i = 10
    a = aggs.Boxplot(field='load_time')
    assert {'boxplot': {'field': 'load_time'}} == a.to_dict()

def test_rare_terms_aggregation():
    if False:
        print('Hello World!')
    a = aggs.RareTerms(field='the-field')
    a.bucket('total_sales', 'sum', field='price')
    a.bucket('sales_bucket_sort', 'bucket_sort', sort=[{'total_sales': {'order': 'desc'}}], size=3)
    assert {'aggs': {'sales_bucket_sort': {'bucket_sort': {'size': 3, 'sort': [{'total_sales': {'order': 'desc'}}]}}, 'total_sales': {'sum': {'field': 'price'}}}, 'rare_terms': {'field': 'the-field'}} == a.to_dict()

def test_variable_width_histogram_aggregation():
    if False:
        print('Hello World!')
    a = aggs.VariableWidthHistogram(field='price', buckets=2)
    assert {'variable_width_histogram': {'buckets': 2, 'field': 'price'}} == a.to_dict()

def test_multi_terms_aggregation():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.MultiTerms(terms=[{'field': 'tags'}, {'field': 'author.row'}])
    assert {'multi_terms': {'terms': [{'field': 'tags'}, {'field': 'author.row'}]}} == a.to_dict()

def test_median_absolute_deviation_aggregation():
    if False:
        return 10
    a = aggs.MedianAbsoluteDeviation(field='rating')
    assert {'median_absolute_deviation': {'field': 'rating'}} == a.to_dict()

def test_t_test_aggregation():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.TTest(a={'field': 'startup_time_before'}, b={'field': 'startup_time_after'}, type='paired')
    assert {'t_test': {'a': {'field': 'startup_time_before'}, 'b': {'field': 'startup_time_after'}, 'type': 'paired'}} == a.to_dict()

def test_inference_aggregation():
    if False:
        for i in range(10):
            print('nop')
    a = aggs.Inference(model_id='model-id', buckets_path={'agg_name': 'agg_name'})
    assert {'inference': {'buckets_path': {'agg_name': 'agg_name'}, 'model_id': 'model-id'}} == a.to_dict()

def test_moving_percentiles_aggregation():
    if False:
        i = 10
        return i + 15
    a = aggs.DateHistogram()
    a.bucket('the_percentile', 'percentiles', field='price', percents=[1.0, 99.0])
    a.pipeline('the_movperc', 'moving_percentiles', buckets_path='the_percentile', window=10)
    assert {'aggs': {'the_movperc': {'moving_percentiles': {'buckets_path': 'the_percentile', 'window': 10}}, 'the_percentile': {'percentiles': {'field': 'price', 'percents': [1.0, 99.0]}}}, 'date_histogram': {}} == a.to_dict()

def test_normalize_aggregation():
    if False:
        return 10
    a = aggs.Normalize(buckets_path='normalized', method='percent_of_sum')
    assert {'normalize': {'buckets_path': 'normalized', 'method': 'percent_of_sum'}} == a.to_dict()