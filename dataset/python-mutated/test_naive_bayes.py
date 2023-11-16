from __future__ import annotations
import pandas as pd
import pytest
from sklearn import naive_bayes as sk_naive_bayes
from river import compose, feature_extraction, naive_bayes

def river_models():
    if False:
        i = 10
        return i + 15
    'List of Naive Bayes models to test.'
    yield from [naive_bayes.MultinomialNB, naive_bayes.BernoulliNB, naive_bayes.ComplementNB]

def sklearn_models():
    if False:
        for i in range(10):
            print('nop')
    'Mapping between Naives Bayes river models and sklearn models.'
    yield from [(naive_bayes.MultinomialNB, sk_naive_bayes.MultinomialNB), (naive_bayes.BernoulliNB, sk_naive_bayes.BernoulliNB), (naive_bayes.ComplementNB, sk_naive_bayes.ComplementNB)]

def yield_dataset():
    if False:
        print('Hello World!')
    'Incremental dataset.'
    yield from [('Chinese Beijing Chinese', 'yes'), ('Chinese Chinese Shanghai', 'yes'), ('Chinese Macao', 'yes'), ('Tokyo Japan Chinese', 'no')]

def yield_batch_dataset():
    if False:
        i = 10
        return i + 15
    'Batch dataset.'
    for (x, y) in yield_dataset():
        yield (pd.Series([x]), pd.Series([y]))

def yield_unseen_data():
    if False:
        while True:
            i = 10
    yield from ['Chinese Beijing Chinese', 'Chinese Chinese Shanghai', 'Chinese Macao', 'Tokyo Japan Chinese', 'new unseen data', 'Taiwanese Taipei', 'Chinese ShanghaiShanghai', 'Chinese', 'Tokyo Macao', 'Tokyo Tokyo', 'Macao Macao new', 'new']

def yield_batch_unseen_data():
    if False:
        return 10
    yield from [pd.Series(x) for x in yield_unseen_data()]

@pytest.mark.parametrize('model', [pytest.param(compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)), ('model', model(alpha=alpha))), id=f'{model.__name__} - {alpha}') for model in river_models() for alpha in [alpha for alpha in range(1, 4)]])
def test_learn_one_methods(model):
    if False:
        while True:
            i = 10
    'Assert that the methods of the Naives Bayes class behave correctly.'
    assert model.predict_proba_one('not fitted yet') == {}
    assert model.predict_one('not fitted yet') is None
    for (x, y) in yield_dataset():
        model = model.learn_one(x, y)
    if isinstance(model['model'], naive_bayes.ComplementNB) or isinstance(model['model'], naive_bayes.MultinomialNB):
        assert model['model']._more_tags() == {'positive input'}
    assert model['model']._multiclass

@pytest.mark.parametrize('model, batch_model', [pytest.param(compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)), ('model', model(alpha=alpha))), compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)), ('model', model(alpha=alpha))), id=f'{model.__name__} - {alpha}') for model in river_models() for alpha in [alpha for alpha in range(1, 4)]])
def test_learn_many_vs_learn_one(model, batch_model):
    if False:
        return 10
    'Assert that the Naive Bayes river models provide the same results when learning in\n    incremental and mini-batch modes. The models tested are MultinomialNB, BernoulliNB and\n    ComplementNB with differents alpha parameters..\n    '
    for (x, y) in yield_dataset():
        model = model.learn_one(x, y)
    for (x, y) in yield_batch_dataset():
        batch_model = batch_model.learn_many(x, y)
    assert model['model'].p_class('yes') == batch_model['model'].p_class('yes')
    assert model['model'].p_class('no') == batch_model['model'].p_class('no')
    for (x, x_batch) in zip(yield_unseen_data(), yield_batch_unseen_data()):
        assert model.predict_proba_one(x)['yes'] == pytest.approx(batch_model.predict_proba_many(x_batch)['yes'][0])
        assert model.predict_proba_one(x)['no'] == pytest.approx(batch_model.predict_proba_many(x_batch)['no'][0])
    assert model['model'].p_class('yes') == batch_model['model'].p_class('yes')
    assert model['model'].p_class('no') == batch_model['model'].p_class('no')
    if isinstance(model['model'], naive_bayes.BernoulliNB) or isinstance(model['model'], naive_bayes.MultinomialNB):
        inc_cp = model['model'].p_feature_given_class
        batch_cp = batch_model['model'].p_feature_given_class
        assert inc_cp('Chinese', 'yes') == batch_cp('Chinese', 'yes')
        assert inc_cp('Tokyo', 'yes') == batch_cp('Tokyo', 'yes')
        assert inc_cp('Japan', 'yes') == batch_cp('Japan', 'yes')
        assert inc_cp('Chinese', 'no') == batch_cp('Chinese', 'no')
        assert inc_cp('Tokyo', 'no') == batch_cp('Tokyo', 'no')
        assert inc_cp('Japan', 'no') == batch_cp('Japan', 'no')
        assert inc_cp('unseen', 'yes') == batch_cp('unseen', 'yes')
    assert model['model'].class_counts == batch_model['model'].class_counts
    assert model['model'].feature_counts == batch_model['model'].feature_counts
    if isinstance(model['model'], naive_bayes.ComplementNB) or isinstance(model['model'], naive_bayes.MultinomialNB):
        assert model['model'].class_totals == batch_model['model'].class_totals
    if isinstance(model['model'], sk_naive_bayes.ComplementNB):
        assert model['model'].feature_totals == batch_model['model'].feature_totals

@pytest.mark.parametrize('batch_model', [pytest.param(compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)), ('model', model(alpha=alpha))), id=f'{model.__name__} - {alpha}') for model in river_models() for alpha in [alpha for alpha in range(1, 4)]])
def test_learn_many_not_fit(batch_model):
    if False:
        i = 10
        return i + 15
    'Ensure that Naives Bayes models return an empty DataFrame when not yet fitted. Also check\n    that the predict_proba_many method keeps the index.\n    '
    assert batch_model.predict_proba_many(pd.Series(['new', 'unseen'], index=['river', 'rocks'])).equals(pd.DataFrame(index=['river', 'rocks']))
    assert batch_model.predict_many(pd.Series(['new', 'unseen'], index=['river', 'rocks'])).equals(pd.DataFrame(index=['river', 'rocks']))

@pytest.mark.parametrize('model, sk_model, bag', [pytest.param(compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)), ('model', model(alpha=alpha))), sk_model(alpha=alpha), feature_extraction.BagOfWords(lowercase=False), id=f'{model.__name__} - {alpha}') for (model, sk_model) in sklearn_models() for alpha in [alpha for alpha in range(1, 4)]])
def test_river_vs_sklearn(model, sk_model, bag):
    if False:
        print('Hello World!')
    'Assert that river Naive Bayes models and sklearn Naive Bayes models provide the same results\n    when the input data are the same. Also check that the behaviour of Naives Bayes models are the\n    same with dense and sparse dataframe. Models tested are MultinomialNB, BernoulliNB and\n    ComplementNB with differents alpha parameters.\n    '
    for (x, y) in yield_batch_dataset():
        model = model.learn_many(x, y)
    X = pd.concat([x for (x, _) in yield_batch_dataset()])
    y = pd.concat([y for (_, y) in yield_batch_dataset()])
    sk_model = sk_model.fit(X=bag.transform_many(X), y=y)
    for (sk_preds, river_preds) in zip(sk_model.predict_proba(bag.transform_many(X)), model.predict_proba_many(X).values):
        for (sk_pred, river_pred) in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(1 - sk_pred) or river_pred == pytest.approx(sk_pred)
    for (sk_preds, river_preds) in zip(sk_model.predict_proba(bag.transform_many(X).sparse.to_dense()), model['model'].predict_proba_many(bag.transform_many(X).sparse.to_dense()).values):
        for (sk_pred, river_pred) in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(1 - sk_pred) or river_pred == pytest.approx(sk_pred)