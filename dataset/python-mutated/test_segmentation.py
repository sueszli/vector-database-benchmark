"""
Scripts to test cleanlab.segmentation package
"""
import numpy as np
import numpy as np
import random
np.random.seed(0)
import pytest
from unittest import mock
import matplotlib.pyplot as plt
from cleanlab.internal.multilabel_scorer import softmin
from cleanlab.internal.segmentation_utils import _check_input, _get_valid_optional_params, _get_summary_optional_params
from cleanlab.segmentation.filter import find_label_issues
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores, _get_label_quality_per_image
from cleanlab.segmentation.summary import display_issues, common_label_issues, filter_by_class, _generate_colormap

def generate_three_image_dataset(bad_index):
    if False:
        while True:
            i = 10
    good_gt = np.zeros((10, 10))
    good_gt[:5, :] = 1.0
    bad_gt = np.ones((10, 10))
    bad_gt[:5, :] = 0.0
    good_pr = np.random.random((2, 10, 10))
    good_pr[0, :5, :] = good_pr[0, :5, :] / 10
    good_pr[1, 5:, :] = good_pr[1, 5:, :] / 10
    val = np.binary_repr([4, 2, 1][bad_index], width=3)
    error = [int(case) for case in val]
    labels = []
    pred = []
    for case in val:
        if case == '0':
            labels.append(good_gt)
            pred.append(good_pr)
        else:
            labels.append(bad_gt)
            pred.append(good_pr)
    labels = np.array(labels)
    pred_probs = np.array(pred)
    return (labels, pred_probs, error)
(labels, pred_probs, error) = generate_three_image_dataset(random.randint(0, 2))
(labels, pred_probs) = (labels.astype(int), pred_probs.astype(float))
(num_images, num_classes, h, w) = pred_probs.shape

def test_find_label_issues():
    if False:
        print('Hello World!')
    issues = find_label_issues(labels, pred_probs, n_jobs=None, batch_size=1000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    issues = find_label_issues(labels, pred_probs, downsample=2, batch_size=1739)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    issues = find_label_issues(labels, pred_probs, downsample=5, n_jobs=None, batch_size=2838)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    with pytest.raises(Exception) as e:
        issues = find_label_issues(labels, pred_probs, downsample=4, n_jobs=None, batch_size=1000)
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=2000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=500)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    issues = find_label_issues(labels, pred_probs, downsample=1, verbose=False)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    with pytest.raises(Exception) as e:
        issues = find_label_issues(labels, pred_probs, downsample=3, n_jobs=None, batch_size=1000)
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=2, batch_size=1000)
    assert np.argmax(error) == np.argmax(issues.sum((1, 2)))
    with pytest.raises(Exception) as e:
        issues = find_label_issues(np.array([[[[1, 2, 3]]]]), pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    with pytest.raises(Exception) as e:
        issues = find_label_issues(labels, np.array([[[[0.1, 0.2, 0.3]]]]), downsample=1, n_jobs=None, batch_size=1000)

def test_find_label_issues_sizes():
    if False:
        for i in range(10):
            print('nop')
    (labels, pred_probs) = (np.random.randint(0, 2, (2, 9, 7)), np.random.random((2, 2, 9, 7)))
    issues = find_label_issues(labels, pred_probs)
    (labels, pred_probs) = (np.random.randint(0, 2, (2, 13, 47)), np.random.random((2, 2, 13, 47)))
    issues = find_label_issues(labels, pred_probs)
    for _ in range(5):
        (h, w) = np.random.randint(1, 100, 2)
        (labels, pred_probs) = (np.random.randint(0, 2, (2, h, w)), np.random.random((2, 2, h, w)))
        issues = find_label_issues(labels, pred_probs)

def test__check_input():
    if False:
        return 10
    bad_gt = np.random.random((5, 10, 20))
    with pytest.raises(Exception) as e:
        _check_input(bad_gt, bad_gt)
    bad_pr = np.random.random((5, 2, 10, 20))
    with pytest.raises(Exception) as e:
        _check_input(bad_pr, bad_pr)
    smaller_pr = np.random.random((5, 2, 9, 20))
    with pytest.raises(Exception) as e:
        _check_input(bad_gt, smaller_pr)
    fewer_gt = np.random.random((4, 10, 20))
    with pytest.raises(Exception) as e:
        _check_input(fewer_gt, smaller_pr)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_get_label_quality_scores():
    if False:
        return 10
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    assert np.argmax(error) == np.argmin(image_scores_softmin)
    with pytest.raises(Exception) as e:
        get_label_quality_scores(labels, pred_probs, method='num_pixel_issues', downsample=4)
    (image_scores_npi, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='num_pixel_issues', downsample=1)
    assert np.argmax(error) == np.argmin(image_scores_npi)
    with pytest.raises(Exception):
        get_label_quality_scores(labels, pred_probs, method='invalid_method')
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, downsample=1, method='softmin')
    assert len(image_scores_softmin) == labels.shape[0]
    assert pixel_scores.shape == labels.shape
    with pytest.raises(ValueError):
        get_label_quality_scores(labels, pred_probs, method='num_pixel_issues', batch_size=-1)
        get_label_quality_scores(labels, pred_probs, method='num_pixel_issues', downsample=1, batch_size=0)

def test_get_label_quality_scores_sizes():
    if False:
        for i in range(10):
            print('nop')
    (labels, pred_probs) = (np.random.randint(0, 2, (2, 9, 7)), np.random.random((2, 2, 9, 7)))
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    (labels, pred_probs) = (np.random.randint(0, 2, (2, 13, 47)), np.random.random((2, 2, 13, 47)))
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    for _ in range(5):
        (h, w) = np.random.randint(1, 100, 2)
        (labels, pred_probs) = (np.random.randint(0, 2, (2, h, w)), np.random.random((2, 2, h, w)))
        (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')

def test_issues_from_scores():
    if False:
        print('Hello World!')
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    issues_from_score = issues_from_scores(image_scores_softmin, pixel_scores, threshold=1)
    assert np.shape(issues_from_score) == np.shape(pixel_scores)
    assert h * w * num_images == issues_from_score.sum()
    issues_from_score = issues_from_scores(image_scores_softmin, pixel_scores, threshold=0)
    assert 0 == issues_from_score.sum()
    issues_from_score = issues_from_scores(image_scores_softmin, pixel_scores, threshold=0.5)
    assert np.argmax(error) == np.argmax(issues_from_score.sum((1, 2)))
    sort_by_score = issues_from_scores(image_scores_softmin, threshold=0.5)
    assert error[sort_by_score[0]] == 1

def test_issues_from_scores_no_pixel_scores():
    if False:
        print('Hello World!')
    (image_scores_softmin, _) = get_label_quality_scores(labels, pred_probs, method='softmin')
    issues_from_score_result = issues_from_scores(image_scores_softmin, None, threshold=1)
    assert np.shape(issues_from_score_result) == (num_images,)

def test_issues_from_scores_various_thresholds():
    if False:
        i = 10
        return i + 15
    (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    for threshold in [0.1, 0.5, 0.9]:
        issues_from_score_result = issues_from_scores(image_scores_softmin, pixel_scores, threshold=threshold)
        assert np.all(issues_from_score_result == (pixel_scores < threshold))

def test_issues_from_scores_invalid_inputs():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        issues_from_scores(None)
    with pytest.raises(ValueError):
        issues_from_scores(np.array([0.1, 0.2, 0.3]), threshold=1.1)
    with pytest.raises(ValueError):
        issues_from_scores(np.array([0.1, 0.2, 0.3]), threshold=-0.1)

def test_issues_from_scores_different_input_sizes():
    if False:
        print('Hello World!')
    for num_images in range(1, 5):
        image_scores = np.random.rand(num_images)
        pixel_scores = np.random.rand(num_images, 100, 100)
        issues_from_score_result = issues_from_scores(image_scores, pixel_scores, threshold=0.5)
        assert np.shape(issues_from_score_result) == np.shape(pixel_scores)

def test_issues_from_scores_sorting():
    if False:
        i = 10
        return i + 15
    (image_scores_softmin, _) = get_label_quality_scores(labels, pred_probs, method='softmin')
    issues_from_score_result = issues_from_scores(image_scores_softmin, None, threshold=0.5)
    assert np.all(np.sort(image_scores_softmin) == image_scores_softmin[issues_from_score_result])

def test__get_label_quality_per_image():
    if False:
        return 10
    random_score_array = np.random.random((100,))
    temp = random.random()
    score = _get_label_quality_per_image(random_score_array, method='softmin', temperature=temp)
    cleanlab_softmin = softmin(np.expand_dims(random_score_array, axis=0), axis=1, temperature=temp)[0]
    assert cleanlab_softmin == score, 'Expected cleanlab_softmin to be equal to score'
    empty_score_array = np.array([])
    with pytest.raises(Exception) as e:
        _get_label_quality_per_image(empty_score_array, method='softmin', temperature=temp)
    with pytest.raises(Exception):
        _get_label_quality_per_image(random_score_array, method=None, temperature=temp)
    with pytest.raises(Exception):
        _get_label_quality_per_image(random_score_array, method='invalid_method', temperature=temp)
    with pytest.raises(Exception):
        _get_label_quality_per_image(random_score_array, method='softmin', temperature=0)
    with pytest.raises(Exception):
        _get_label_quality_per_image(random_score_array, method='softmin', temperature=None)

def test_generate_color_map():
    if False:
        while True:
            i = 10
    colors = _generate_colormap(0)
    assert len(colors) == 0
    colors = _generate_colormap(1)
    assert len(colors) == 1
    assert len(colors[0]) == 4
    colors = _generate_colormap(-1)
    assert len(colors) == 0
    colors = _generate_colormap(5)
    assert len(colors) == 5
    num_colors = 385
    colors = _generate_colormap(num_colors)
    unique_rows = np.unique(colors, axis=0)
    assert unique_rows.shape[0] == num_colors

def test_display_issues(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(plt, 'show', lambda : None)
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    display_issues(issues, top=1)
    display_issues(issues, pred_probs=pred_probs, labels=labels, top=2, class_names=['one', 'two'])
    display_issues(issues, pred_probs=pred_probs, labels=labels, top=2)
    display_issues(issues, labels=labels, top=2)
    display_issues(issues, pred_probs=pred_probs, top=2)
    display_issues(issues, pred_probs=pred_probs, labels=labels, top=len(issues) + 5)
    class_issues = filter_by_class(0, issues, labels=labels, pred_probs=pred_probs)
    display_issues(class_issues, pred_probs=pred_probs, labels=labels, top=2)
    (image_scores, pixel_scores) = (image_scores_softmin, pixel_scores) = get_label_quality_scores(labels, pred_probs, method='softmin')
    issues_from_score = issues_from_scores(image_scores, pixel_scores, threshold=0.5)
    display_issues(issues_from_score, pred_probs=pred_probs, labels=labels, top=2)
    display_issues(issues_from_score, pred_probs=pred_probs, labels=labels, top=2, exclude=[0])
    with pytest.raises(ValueError) as e:
        display_issues(issues_from_score, pred_probs=pred_probs, labels=None, top=2, exclude=[0])

@mock.patch('matplotlib.pyplot.figure')
def test_display_issues_figure(mock_plt, monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(plt, 'show', lambda : None)
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    display_issues(issues, pred_probs=pred_probs, labels=labels, top=2, class_names=['one', 'two'])
    assert mock_plt.called

@mock.patch('matplotlib.pyplot.show')
def test_display_issues_show(mock_plt):
    if False:
        i = 10
        return i + 15
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    display_issues(issues, top=1)
    assert mock_plt.called

def test_common_label_issues(capsys):
    if False:
        for i in range(10):
            print('nop')
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    df = common_label_issues(issues, labels, pred_probs)
    df_without_0 = common_label_issues(issues, labels, pred_probs, exclude=[0])
    assert df.shape != df_without_0.shape
    captured_words = capsys.readouterr()
    df = common_label_issues(issues, labels, pred_probs, verbose=False)
    captured_no_words = capsys.readouterr()
    assert len(captured_no_words.out) == 0
    assert len(captured_words.out) > 0
    df_class_names = common_label_issues(issues, labels, pred_probs, class_names=['one', 'two'])
    captured_top_all = capsys.readouterr()
    df_top_1 = common_label_issues(issues, labels, pred_probs, top=1)
    captured_top_1 = capsys.readouterr()
    assert len(captured_top_1.out) < len(captured_top_all.out)
    assert df_class_names['given_label'].to_list() != df['given_label'].to_list()

def test_filter_by_class():
    if False:
        while True:
            i = 10
    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
    class_0_issues = filter_by_class(0, issues, labels=labels, pred_probs=pred_probs)
    class_1_issues = filter_by_class(1, issues, labels=labels, pred_probs=pred_probs)
    class_300_issues = filter_by_class(300, issues, labels=labels, pred_probs=pred_probs)
    assert (class_0_issues == class_1_issues).all()
    assert np.sum(class_300_issues) == 0

def test_summary_sizes(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(plt, 'show', lambda : None)
    for _ in range(5):
        (h, w) = np.random.randint(1, 100, 2)
        (labels, pred_probs) = (np.random.randint(0, 2, (2, h, w)), np.random.random((2, 2, h, w)))
        issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=1000)
        class_300_issues = filter_by_class(0, issues, labels=labels, pred_probs=pred_probs)
        df = common_label_issues(issues, labels, pred_probs)
        display_issues(issues)

def test_get_valid_functions():
    if False:
        print('Hello World!')
    optional_batch_size = 10
    optional_n_jobs = 2
    (x, y) = _get_valid_optional_params(optional_batch_size, optional_n_jobs)
    assert x == optional_batch_size and y == optional_n_jobs
    (x, y) = _get_valid_optional_params(None, None)
    assert x == 10000 and y == None
    optional_class_names = [1, 2]
    optional_exclude = [1]
    optional_top = 10
    (x, y, z) = _get_summary_optional_params(optional_class_names, optional_exclude, optional_top)
    assert x == optional_class_names and y == optional_exclude and (z == optional_top)
    (x, y, z) = _get_summary_optional_params(None, None, None)
    assert x == None and y == [] and (z == 20)