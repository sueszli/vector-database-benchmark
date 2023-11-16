import itertools
import typing
import numpy as np
import pytest
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from cleanlab.internal import multilabel_scorer as ml_scorer
from cleanlab.internal.multilabel_utils import stack_complement, get_onehot_num_classes, onehot2int
from cleanlab import multilabel_classification as ml_classification
from cleanlab.multilabel_classification.dataset import common_multilabel_issues, rank_classes_by_multilabel_quality, overall_multilabel_health_score, multilabel_health_summary
from cleanlab.multilabel_classification.rank import get_label_quality_scores_per_class
from cleanlab.multilabel_classification import filter

@pytest.fixture
def labels():
    if False:
        while True:
            i = 10
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]])

@pytest.fixture
def pred_probs_gold(labels):
    if False:
        print('Hello World!')
    pred_probs = np.array([[0.203, 0.465, 0.612], [0.802, 0.596, 0.43], [0.776, 0.649, 0.391], [0.201, 0.439, 0.633], [0.203, 0.443, 0.584], [0.814, 0.572, 0.332], [0.201, 0.388, 0.544], [0.778, 0.646, 0.392], [0.796, 0.611, 0.387], [0.199, 0.381, 0.58]])
    assert pred_probs.shape == labels.shape
    return pred_probs

@pytest.fixture
def pred_probs():
    if False:
        print('Hello World!')
    return np.array([[0.9, 0.1, 0.2], [0.5, 0.6, 0.4], [0.75, 0.8, 0.85], [0.9, 0.85, 0.2], [0.9, 0.1, 0.85], [0.5, 0.6, 0.85], [0.9, 0.85, 0.85], [0.8, 0.4, 0.2], [0.9, 0.1, 0.85], [0.15, 0.95, 0.05]])

@pytest.fixture
def pred_probs_multilabel():
    if False:
        return 10
    return np.array([[0.9, 0.1, 0.0, 0.4, 0.1], [0.7, 0.8, 0.2, 0.3, 0.1], [0.9, 0.8, 0.4, 0.2, 0.1], [0.1, 0.1, 0.8, 0.3, 0.1], [0.4, 0.5, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.8, 0.1, 0.2, 0.1, 0.1]])

@pytest.fixture
def labels_multilabel():
    if False:
        return 10
    return [[0], [0, 1], [0, 1], [2], [0, 2, 3], [], []]

@pytest.fixture
def data_multilabel(num_classes=5):
    if False:
        for i in range(10):
            print('nop')
    labels = []
    pred_probs = []
    for i in range(0, 100):
        q = [0.1] * num_classes
        pos = i % num_classes
        labels.append([pos])
        if i > 90:
            pos = (pos + 2) % num_classes
        q[pos] = 0.9
        pred_probs.append(q)
    return (labels, np.array(pred_probs))

@pytest.fixture
def cv():
    if False:
        i = 10
        return i + 15
    return sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

@pytest.fixture
def dummy_features(labels):
    if False:
        i = 10
        return i + 15
    np.random.seed(42)
    return np.random.rand(labels.shape[0], 2)

def test_public_label_quality_scores(labels, pred_probs):
    if False:
        i = 10
        return i + 15
    formatted_labels = onehot2int(labels)
    assert isinstance(formatted_labels, list)
    scores1 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs)
    assert len(scores1) == len(labels)
    assert (scores1 >= 0).all() and (scores1 <= 1).all()
    scores2 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='confidence_weighted_entropy')
    assert not np.isclose(scores1, scores2).all()
    scores3 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, adjust_pred_probs=True)
    assert not np.isclose(scores1, scores3).all()
    scores4 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='normalized_margin', adjust_pred_probs=True, aggregator_kwargs={'method': 'exponential_moving_average'})
    assert not np.isclose(scores1, scores4).all()
    scores5 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='normalized_margin', adjust_pred_probs=True, aggregator_kwargs={'method': 'softmin'})
    assert not np.isclose(scores4, scores5).all()
    scores6 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='normalized_margin', adjust_pred_probs=True, aggregator_kwargs={'method': 'softmin', 'temperature': 0.002})
    assert not np.isclose(scores5, scores6).all()
    scores7 = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='normalized_margin', adjust_pred_probs=True, aggregator_kwargs={'method': np.min})
    assert np.isclose(scores6, scores7, rtol=0.001).all()
    with pytest.raises(ValueError) as e:
        _ = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, method='badchoice')
        assert 'Invalid method name: badchoice' in str(e.value)
    with pytest.raises(ValueError) as e:
        _ = ml_classification.get_label_quality_scores(formatted_labels, pred_probs, aggregator_kwargs={'method': 'invalid'})
        assert "Invalid aggregation method specified: 'invalid'" in str(e.value)

class TestAggregator:
    """Test the Aggregator class."""

    @pytest.fixture
    def base_scores(self):
        if False:
            i = 10
            return i + 15
        return np.array([[0.6, 0.3, 0.7, 0.1, 0.9]])

    @pytest.mark.parametrize('method', [np.min, np.max, np.mean, np.median, 'exponential_moving_average', 'softmin'], ids=lambda x: x.__name__ if callable(x) else str(x))
    def test_aggregator_callable(self, method):
        if False:
            i = 10
            return i + 15
        aggregator = ml_scorer.Aggregator(method=method)
        assert callable(aggregator.method), 'Aggregator should store a callable method'
        assert callable(aggregator), 'Aggregator should be callable'

    @pytest.mark.parametrize('method,expected_score', [(np.min, 0.1), (np.max, 0.9), (np.mean, 0.52), (np.median, 0.6), ('exponential_moving_average', 0.436), ('softmin', 0.128)], ids=['min', 'max', 'mean', 'median', 'exponential_moving_average', 'softmin'])
    def test_aggregator_score(self, base_scores, method, expected_score):
        if False:
            while True:
                i = 10
        aggregator = ml_scorer.Aggregator(method=method)
        scores = aggregator(base_scores)
        assert np.isclose(scores, np.array([expected_score]), rtol=0.001).all()
        assert scores.shape == (1,)

    def test_invalid_method(self):
        if False:
            return 10
        with pytest.raises(ValueError) as e:
            _ = ml_scorer.Aggregator(method='invalid_method')
            assert "Invalid aggregation method specified: 'invalid_method'" in str(e.value), 'String constructor has limited options'
        with pytest.raises(TypeError) as e:
            _ = ml_scorer.Aggregator(method=1)
            assert 'Expected callable method' in str(e.value), 'Non-callable methods are not valid'

    def test_invalid_score(self, base_scores):
        if False:
            print('Hello World!')
        aggregator = ml_scorer.Aggregator(method=np.min)
        with pytest.raises(ValueError) as e:
            _ = aggregator(base_scores[0])
            assert 'Expected 2D array' in str(e.value), 'Aggregator expects 2D array'

class TestMultilabelScorer:
    """Test the MultilabelScorer class."""

    @pytest.fixture
    def docs_labels(self):
        if False:
            print('Hello World!')
        return np.array([[0, 1, 0], [1, 0, 1]])

    @pytest.fixture
    def docs_pred_probs(self):
        if False:
            while True:
                i = 10
        return np.array([[0.1, 0.9, 0.7], [0.4, 0.1, 0.6]])

    @pytest.fixture
    def default_scorer(self):
        if False:
            return 10
        return ml_scorer.MultilabelScorer()

    @pytest.mark.parametrize('base_scorer', [scorer for scorer in ml_scorer.ClassLabelScorer], ids=lambda x: x.name)
    @pytest.mark.parametrize('aggregator', [np.min, np.max, np.mean, 'exponential_moving_average', 'softmin'])
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', ''])
    def test_call(self, base_scorer, aggregator, strict, labels, pred_probs):
        if False:
            i = 10
            return i + 15
        scorer = ml_scorer.MultilabelScorer(base_scorer, aggregator, strict=strict)
        assert callable(scorer)
        test_scores = scorer(labels, pred_probs)
        assert isinstance(test_scores, np.ndarray)
        assert test_scores.shape == (labels.shape[0],)
        base_scorer_kwargs = {'adjust_pred_probs': True}
        if scorer.base_scorer is not ml_scorer.ClassLabelScorer.CONFIDENCE_WEIGHTED_ENTROPY:
            test_scores = scorer(labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs)
            assert isinstance(test_scores, np.ndarray)
            assert test_scores.shape == (labels.shape[0],)
        else:
            with pytest.raises(ValueError) as e:
                scorer(labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs)
                assert 'adjust_pred_probs is not currently supported for' in str(e)

    @pytest.mark.parametrize('base_scorer', [scorer for scorer in ml_scorer.ClassLabelScorer], ids=lambda x: x.name)
    def test_aggregate_kwargs(self, base_scorer):
        if False:
            return 10
        'Make sure the instatiated aggregator kwargs can be overridden.\n        I.e. switching from a forgetting-factor 1.0 to 0.5.\n        '
        class_label_quality_scores = np.array([[0.9, 0.9, 0.3], [0.4, 0.9, 0.6]])
        aggregator = ml_scorer.Aggregator(ml_scorer.exponential_moving_average, alpha=1.0)
        scorer = ml_scorer.MultilabelScorer(base_scorer=base_scorer, aggregator=aggregator)
        scores = scorer.aggregate(class_label_quality_scores)
        assert np.allclose(scores, np.array([0.3, 0.4]))
        new_scores = scorer.aggregate(class_label_quality_scores, alpha=0.0)
        assert np.allclose(new_scores, np.array([0.9, 0.9]))

    def test_get_class_label_quality_scores(self, default_scorer, docs_labels, docs_pred_probs):
        if False:
            for i in range(10):
                print('nop')
        'Test the get_class_label_quality_scores method.'
        class_label_quality_scores = default_scorer.get_class_label_quality_scores(docs_labels, docs_pred_probs)
        assert np.allclose(class_label_quality_scores, np.array([[0.9, 0.9, 0.3], [0.4, 0.9, 0.6]]))

@pytest.mark.parametrize('method', ['self_confidence', 'normalized_margin', 'confidence_weighted_entropy'])
def test_class_label_scorer_from_str(method):
    if False:
        return 10
    for m in (method, method.upper()):
        scorer = ml_scorer.ClassLabelScorer.from_str(m)
        assert callable(scorer)
        with pytest.raises(ValueError):
            ml_scorer.ClassLabelScorer.from_str(m.replace('_', '-'))

@pytest.fixture
def scorer():
    if False:
        return 10
    return ml_scorer.MultilabelScorer(base_scorer=ml_scorer.ClassLabelScorer.SELF_CONFIDENCE, aggregator=np.min)

def test_is_multilabel(labels):
    if False:
        i = 10
        return i + 15
    assert ml_scorer._is_multilabel(labels)
    assert not ml_scorer._is_multilabel(labels[:, 0])

@pytest.mark.parametrize('class_names', [None, ['Apple', 'Cat', 'Dog', 'Peach', 'Bird']])
def test_common_multilabel_issues(class_names, pred_probs_multilabel, labels_multilabel):
    if False:
        for i in range(10):
            print('nop')
    df = common_multilabel_issues(labels=labels_multilabel, pred_probs=pred_probs_multilabel, class_names=class_names)
    expected_issue_probabilities = [0.14285714285714285, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert len(df) == 10
    assert np.isclose(np.array(expected_issue_probabilities), df['Issue Probability']).all()
    if class_names:
        expected_res = ['Apple', 'Dog', 'Apple', 'Cat', 'Cat', 'Dog', 'Peach', 'Peach', 'Bird', 'Bird']
        assert list(df['Class Name']) == expected_res
    else:
        assert 'Class Name' not in df.columns

def test_multilabel_find_label_issues(data_multilabel):
    if False:
        while True:
            i = 10
    (labels, pred_probs) = data_multilabel
    issues = filter.find_label_issues(labels=labels, pred_probs=pred_probs, return_indices_ranked_by='self_confidence')
    issues_lm = filter.find_label_issues(labels, pred_probs, low_memory=True, return_indices_ranked_by='self_confidence')
    intersection = len(list(set(issues).intersection(set(issues_lm))))
    union = len(set(issues)) + len(set(issues_lm)) - intersection
    assert float(intersection) / union > 0.95
    issues_mask = filter.find_label_issues(labels=labels, pred_probs=pred_probs)
    issues_lm_mask = filter.find_label_issues(labels, pred_probs, low_memory=True)
    issues_from_mask = np.where(issues_mask)[0]
    issues_lm_from_mask = np.where(issues_lm_mask)[0]
    intersection = len(list(set(issues_from_mask).intersection(set(issues_lm_from_mask))))
    union = len(set(issues_from_mask)) + len(set(issues_lm_from_mask)) - intersection
    assert float(intersection) / union > 0.95
    rank_by_kwargs = {'adjust_pred_probs': None}
    issues_lm2 = filter.find_label_issues(labels, pred_probs, low_memory=True, return_indices_ranked_by='self_confidence', rank_by_kwargs=rank_by_kwargs, n_jobs=1)
    np.testing.assert_array_equal(issues_lm2, issues_lm)

@pytest.mark.parametrize('min_examples_per_class', [10, 90])
def test_multilabel_min_examples_per_class(data_multilabel, min_examples_per_class):
    if False:
        print('Hello World!')
    (labels, pred_probs) = data_multilabel
    issues = filter.find_label_issues(labels=labels, pred_probs=pred_probs, min_examples_per_class=min_examples_per_class)
    if min_examples_per_class == 10:
        assert sum(issues) == 9
    else:
        assert sum(issues) == 0

@pytest.mark.parametrize('num_to_remove_per_class', [None, [1, 1, 0, 0, 2], [1, 1, 0, 0, 1]])
def test_multilabel_num_to_remove_per_class(data_multilabel, num_to_remove_per_class):
    if False:
        print('Hello World!')
    (labels, pred_probs) = data_multilabel
    issues = filter.find_label_issues(labels=labels, pred_probs=pred_probs, num_to_remove_per_class=num_to_remove_per_class)
    num_issues = sum(issues)
    if num_to_remove_per_class is None:
        assert num_issues == 9
    else:
        assert num_issues == sum(num_to_remove_per_class)

@pytest.mark.parametrize('class_names', [None, ['Apple', 'Cat', 'Dog', 'Peach', 'Bird']])
def test_rank_classes_by_multilabel_quality(pred_probs_multilabel, labels_multilabel, class_names):
    if False:
        for i in range(10):
            print('nop')
    df_ranked = rank_classes_by_multilabel_quality(pred_probs=pred_probs_multilabel, labels=labels_multilabel, class_names=class_names)
    expected_Label_Issues = [1, 0, 0, 0, 0]
    expected_Label_Noise = [0.14285714285714285, 0.0, 0.0, 0.0, 0.0]
    expected_Label_Quality_Score = [0.8571428571428572, 1.0, 1.0, 1.0, 1.0]
    expected_Inverse_Label_Issues = [0, 1, 0, 0, 0]
    expected_Inverse_Label_Noise = [0.0, 0.14285714285714285, 0.0, 0.0, 0.0]
    assert list(df_ranked['Label Issues']) == expected_Label_Issues
    assert np.isclose(np.array(expected_Label_Noise), df_ranked['Label Noise']).all()
    assert np.isclose(np.array(expected_Label_Quality_Score), df_ranked['Label Quality Score']).all()
    assert list(df_ranked['Inverse Label Issues']) == expected_Inverse_Label_Issues
    assert np.isclose(np.array(expected_Inverse_Label_Noise), df_ranked['Inverse Label Noise']).all()
    if class_names:
        expected_res = ['Dog', 'Apple', 'Cat', 'Peach', 'Bird']
        assert list(df_ranked['Class Name']) == expected_res
    else:
        assert 'Class Name' not in df_ranked.columns

def test_overall_multilabel_health_score(data_multilabel):
    if False:
        for i in range(10):
            print('nop')
    (labels, pred_probs) = data_multilabel
    overall_label_health_score = overall_multilabel_health_score(pred_probs=pred_probs, labels=labels)
    assert np.isclose(overall_label_health_score, 0.91)

def test_get_class_label_quality_scores():
    if False:
        print('Hello World!')
    pred_probs = np.array([[0.9, 0.1, 0.0, 0.4, 0.1], [0.7, 0.8, 0.2, 0.3, 0.1], [0.9, 0.8, 0.4, 0.2, 0.1], [0.1, 0.1, 0.8, 0.3, 0.1], [0.4, 0.5, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.8, 0.1, 0.2, 0.1, 0.1]])
    labels = [[0], [0, 1], [0, 1], [2], [0, 2, 3], [], []]
    scores = get_label_quality_scores_per_class(pred_probs=pred_probs, labels=labels)
    expected_res = [[0.9, 0.9, 1.0, 0.6, 0.9], [0.7, 0.8, 0.8, 0.7, 0.9], [0.9, 0.8, 0.6, 0.8, 0.9], [0.9, 0.9, 0.8, 0.7, 0.9], [0.4, 0.5, 0.1, 0.1, 0.9], [0.9, 0.9, 0.8, 0.9, 0.9], [0.2, 0.9, 0.8, 0.9, 0.9]]
    assert np.isclose(scores, np.array(expected_res)).all()

def test_health_summary_multilabel(pred_probs_multilabel, labels_multilabel):
    if False:
        while True:
            i = 10
    health_summary_multilabel = multilabel_health_summary(pred_probs=pred_probs_multilabel, labels=labels_multilabel)
    expected_keys = ['classes_by_multilabel_quality', 'common_multilabel_issues', 'overall_multilabel_health_score']
    assert sorted(health_summary_multilabel.keys()) == expected_keys

@pytest.mark.parametrize('input', [[[0], [1, 2], [0, 2]], [['a', 'b'], ['b']], np.array([[[0, 1], [0, 1]], [[1, 1], [0, 0]]]), 1], ids=['lists of ids', 'lists of strings', '3d array', 'scalar'])
def test_is_multilabel_is_false(input):
    if False:
        return 10
    assert not ml_scorer._is_multilabel(input)

def test_stack_complement():
    if False:
        for i in range(10):
            print('nop')
    pred_probs_class = np.array([0.1, 0.9, 0.3, 0.8])
    pred_probs_extended = stack_complement(pred_probs_class)
    pred_probs_expected = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
    assert np.isclose(pred_probs_extended, pred_probs_expected).all()
    pred_probs_class = np.random.rand(100)
    pred_probs_extended = stack_complement(pred_probs_class)
    assert np.sum(pred_probs_extended, axis=1).all() == 1

@pytest.mark.parametrize('pred_probs_test', (None, pytest.lazy_fixture('pred_probs')), ids=['Without probabilities', 'With probabilities'])
def test_get_onehot_num_classes(labels, pred_probs_test):
    if False:
        print('Hello World!')
    labels_list = [np.nonzero(x)[0].tolist() for x in labels]
    (_, num_classes) = get_onehot_num_classes(labels_list, pred_probs_test)
    assert num_classes == 3

def test_get_label_quality_scores_output(labels, pred_probs, scorer):
    if False:
        i = 10
        return i + 15
    scores = ml_scorer.get_label_quality_scores(labels, pred_probs, method=scorer)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (labels.shape[0],)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.all(np.isfinite(scores))

@pytest.mark.parametrize('given_labels,expected', [(pytest.lazy_fixture('labels'), np.full((3, 2), 0.5)), (np.array([[0, 1], [0, 0], [1, 1]]), np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])), (np.array([[0, 1], [0, 0], [0, 1], [0, 1]]), np.array([[4 / 4, 0 / 4], [1 / 4, 3 / 4]])), (np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]]), np.array([[1, 0] if i != 1 else [0, 1] for i in range(9)]))], ids=['default', 'Missing class assignment configuration', 'Missing class', 'Handle more than 8 classes'])
def test_multilabel_py(given_labels, expected):
    if False:
        while True:
            i = 10
    py = ml_scorer.multilabel_py(given_labels)
    assert isinstance(py, np.ndarray)
    assert py.shape == (given_labels.shape[1], 2)
    assert np.isclose(py, expected).all()

@pytest.mark.parametrize('K', [2, 3, 4], ids=['K=2', 'K=3', 'K=4'])
def test_get_split_generator(cv, K):
    if False:
        return 10
    all_configurations = np.array(list(itertools.product([0, 1], repeat=K)))
    given_labels = np.repeat(all_configurations, 2, axis=0)
    split_generator = ml_scorer._get_split_generator(given_labels, cv)
    assert isinstance(split_generator, typing.Generator)
    (train, test) = next(split_generator)
    for split in (train, test):
        assert isinstance(split, np.ndarray)
        assert np.isin(split, np.arange(given_labels.shape[0])).all()
    (train_labels, test_labels) = (given_labels[train], given_labels[test])
    (_, train_counts) = np.unique(train_labels, axis=0, return_counts=True)
    (_, test_counts) = np.unique(test_labels, axis=0, return_counts=True)
    assert np.all(train_counts == 1)
    assert np.all(test_counts == 1)

@pytest.mark.parametrize('K', [2, 3, 4], ids=['K=2', 'K=3', 'K=4'])
def test_get_split_generator_rare_configurations(cv, K):
    if False:
        return 10
    all_configurations = np.array(list(itertools.product([0, 1], repeat=K)))
    given_labels = np.repeat(all_configurations, 2, axis=0)
    given_labels = given_labels[~np.all(given_labels == all_configurations[0], axis=1)]
    split_generator = ml_scorer._get_split_generator(given_labels, cv)
    (train, test) = next(split_generator)
    (train_labels, test_labels) = (given_labels[train], given_labels[test])
    (_, train_counts) = np.unique(train_labels, axis=0, return_counts=True)
    (_, test_counts) = np.unique(test_labels, axis=0, return_counts=True)
    assert np.all(train_counts == 1)
    assert np.all(test_counts == 1)
    assert len(train_counts) == len(test_counts) == len(all_configurations) - 1
    given_labels = given_labels[1:, :]
    split_generator = ml_scorer._get_split_generator(given_labels, cv)
    (train, test) = next(split_generator)
    (train_labels, test_labels) = (given_labels[train], given_labels[test])
    (_, train_counts) = np.unique(train_labels, axis=0, return_counts=True)
    (_, test_counts) = np.unique(test_labels, axis=0, return_counts=True)
    assert len(train_counts) != len(test_counts)

def test_get_cross_validated_multilabel_pred_probs(dummy_features, labels, cv, pred_probs_gold):
    if False:
        for i in range(10):
            print('nop')
    clf = OneVsRestClassifier(LogisticRegression(random_state=0))
    pred_probs = ml_scorer.get_cross_validated_multilabel_pred_probs(dummy_features, labels, clf=clf, cv=cv)
    assert isinstance(pred_probs, np.ndarray)
    assert pred_probs.shape == labels.shape
    assert np.all(pred_probs >= 0) and np.all(pred_probs <= 1)
    assert np.all(np.isfinite(pred_probs))
    assert dummy_features.shape == (10, 2)
    assert np.allclose(pred_probs, pred_probs_gold, atol=0.0005)

class TestExponentialMovingAverage:
    """Test the ml_scorer.expontential_moving_average function."""

    @pytest.mark.parametrize('alpha', [0.5, None])
    def test_valid_alpha(self, alpha):
        if False:
            i = 10
            return i + 15
        for (x, expected_ema) in zip([np.ones(5).reshape(1, -1), np.array([[0.1, 0.2, 0.3]]), np.array([x / 10 for x in range(1, 7)]).reshape(2, 3)], [1, 0.175, np.array([0.175, 0.475])]):
            ema = ml_scorer.exponential_moving_average(x, alpha=alpha)
            assert np.allclose(ema, expected_ema, atol=0.0001)

    @pytest.mark.parametrize('alpha,expected_ema', [[0, 0.3], [1, 0.1]], ids=['alpha=0', 'alpha=1'])
    def test_alpha_boundary(self, alpha, expected_ema):
        if False:
            for i in range(10):
                print('nop')
        X = np.array([[0.1, 0.2, 0.3]])
        ema = ml_scorer.exponential_moving_average(X, alpha=alpha)
        assert np.allclose(ema, expected_ema, atol=0.0001)

    def test_invalid_alpha(self):
        if False:
            for i in range(10):
                print('nop')
        partial_error_msg = 'alpha must be in the interval \\[0, 1\\]'
        for alpha in [-0.5, 1.5]:
            with pytest.raises(ValueError, match=partial_error_msg):
                ml_scorer.exponential_moving_average(np.ones(5).reshape(1, -1), alpha=alpha)