import numpy as np
import pytest
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, just
from hypothesis import HealthCheck, given, settings
from cleanlab.datalab.internal.issue_manager.null import NullIssueManager
SEED = 42

class TestNullIssueManager:

    @pytest.fixture
    def embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        return embeddings_array

    @pytest.fixture
    def embeddings_with_null(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(SEED)
        embeddings_array = np.random.random((4, 3))
        embeddings_array[0][0] = np.NaN
        embeddings_array[1] = np.NaN
        return embeddings_array

    @pytest.fixture
    def issue_manager(self, lab):
        if False:
            print('Hello World!')
        return NullIssueManager(datalab=lab)

    def test_init(self, lab, issue_manager):
        if False:
            print('Hello World!')
        assert issue_manager.datalab == lab

    def test_find_issues(self, issue_manager, embeddings):
        if False:
            return 10
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        (issues_sort, summary_sort, info_sort) = (issue_manager.issues, issue_manager.summary, issue_manager.info)
        expected_sorted_issue_mask = np.array([False, False, False, False])
        assert np.all(issues_sort['is_null_issue'] == expected_sorted_issue_mask), 'Issue mask should be correct'
        assert summary_sort['issue_type'][0] == 'null'
        assert summary_sort['score'][0] == pytest.approx(expected=1.0, abs=1e-07)
        assert info_sort.get('average_null_score', None) is not None, 'Should have average null score'
        assert summary_sort['score'][0] == pytest.approx(expected=info_sort['average_null_score'], abs=1e-07)

    def test_find_issues_with_null(self, issue_manager, embeddings_with_null):
        if False:
            return 10
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings_with_null)
        (issues_sort, summary_sort, info_sort) = (issue_manager.issues, issue_manager.summary, issue_manager.info)
        expected_sorted_issue_mask = np.array([False, True, False, False])
        assert np.all(issues_sort['is_null_issue'] == expected_sorted_issue_mask), 'Issue mask should be correct'
        assert summary_sort['issue_type'][0] == 'null'
        assert summary_sort['score'][0] == pytest.approx(expected=8 / 12, abs=1e-07)
        assert info_sort.get('average_null_score', None) is not None, 'Should have average null score'
        assert summary_sort['score'][0] == pytest.approx(expected=info_sort['average_null_score'], abs=1e-07)

    def test_report(self, issue_manager, embeddings):
        if False:
            i = 10
            return i + 15
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings)
        report = issue_manager.report(issues=issue_manager.issues, summary=issue_manager.summary, info=issue_manager.info)
        assert isinstance(report, str)
        assert '----------------------- null issues ------------------------\n\nNumber of examples with this issue:' in report
        report = issue_manager.report(issues=issue_manager.issues, summary=issue_manager.summary, info=issue_manager.info, verbosity=3)
        assert 'Additional Information: ' in report

    def test_report_with_null(self, issue_manager, embeddings_with_null):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        issue_manager.find_issues(features=embeddings_with_null)
        report = issue_manager.report(issues=issue_manager.issues, summary=issue_manager.summary, info=issue_manager.info)
        assert isinstance(report, str)
        assert '----------------------- null issues ------------------------\n\nNumber of examples with this issue:' in report
        report = issue_manager.report(issues=issue_manager.issues, summary=issue_manager.summary, info=issue_manager.info, verbosity=3)
        assert 'Additional Information: ' in report

    def test_collect_info(self, issue_manager, embeddings):
        if False:
            print('Hello World!')
        'Test some values in the info dict.'
        issue_manager.find_issues(features=embeddings)
        info = issue_manager.info
        assert info['average_null_score'] == 1.0
        assert info['most_common_issue']['pattern'] == 'no_null'
        assert info['most_common_issue']['count'] == 0
        assert info['most_common_issue']['rows_affected'] == []
        assert info['column_impact'] == [0, 0, 0]

    def test_collect_info_with_nulls(self, issue_manager, embeddings_with_null):
        if False:
            return 10
        'Test some values in the info dict.'
        issue_manager.find_issues(features=embeddings_with_null)
        info = issue_manager.info
        assert info['average_null_score'] == pytest.approx(expected=8 / 12, abs=1e-07)
        assert info['most_common_issue']['pattern'] == '100'
        assert info['most_common_issue']['count'] == 1
        assert info['most_common_issue']['rows_affected'] == [0]
        assert info['column_impact'] == [0.5, 0.25, 0.25]
    nan_strategy = just(np.nan)
    float_with_nan = floats(allow_nan=True)
    features_with_nan_strategy = arrays(dtype=np.float64, shape=array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5), elements=float_with_nan, fill=nan_strategy)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(embeddings=features_with_nan_strategy)
    def test_quality_scores_and_full_null_row_identification(self, issue_manager, embeddings):
        if False:
            print('Hello World!')
        issue_manager.find_issues(features=embeddings)
        (issues_sort, _, _) = (issue_manager.issues, issue_manager.summary, issue_manager.info)
        non_null_fractions = [np.count_nonzero(~np.isnan(row)) / len(row) for row in embeddings]
        scores = issues_sort[issue_manager.issue_score_key]
        assert np.allclose(scores, non_null_fractions, atol=1e-07)
        all_rows_are_null = np.all(np.isnan(embeddings), axis=1)
        assert np.all(issues_sort['is_null_issue'] == all_rows_are_null)