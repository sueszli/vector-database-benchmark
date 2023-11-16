import numpy as np
import pandas as pd
import pytest
from cleanlab.datalab.internal.issue_manager.outlier import OutlierIssueManager
from cleanlab.datalab.internal.issue_manager.data_valuation import DataValuationIssueManager
SEED = 42

class TestDataValuationIssueManager:

    @pytest.fixture
    def issue_manager(self, lab):
        if False:
            print('Hello World!')
        return DataValuationIssueManager(datalab=lab)

    @pytest.fixture
    def outlier_issue_manager(self, lab):
        if False:
            while True:
                i = 10
        return OutlierIssueManager(datalab=lab, k=3)

    @pytest.fixture
    def embeddings(self, lab):
        if False:
            return 10
        np.random.seed(SEED)
        embeddings_array = 0.5 + 0.1 * np.random.rand(lab.get_info('statistics')['num_examples'], 2)
        embeddings_array[4, :] = -1
        return {'embedding': embeddings_array}

    def test_find_issues_with_input(self, issue_manager, embeddings):
        if False:
            i = 10
            return i + 15
        outlier_issue_manager = OutlierIssueManager(datalab=issue_manager.datalab, k=3)
        outlier_issue_manager.find_issues(features=embeddings['embedding'])
        knn_graph = outlier_issue_manager._process_knn_graph_from_features({})
        issue_manager.find_issues(knn_graph=knn_graph)
        (issues, summary, info) = (issue_manager.issues, issue_manager.summary, issue_manager.info)
        assert isinstance(issues, pd.DataFrame), 'Issues should be a dataframe'
        assert isinstance(summary, pd.DataFrame), 'Summary should be a dataframe'
        assert summary['issue_type'].values[0] == 'data_valuation'
        assert isinstance(info, dict), 'Info should be a dict'
        assert isinstance(issues, pd.DataFrame), 'Issues should be a dataframe'
        info_keys = info.keys()
        expected_keys = ['num_low_valuation_issues', 'average_data_valuation']
        assert all([key in info_keys for key in expected_keys]), f'Info should have the right keys, but is missing {set(expected_keys) - set(info_keys)}'

    def test_find_issues_with_stats(self, issue_manager, embeddings):
        if False:
            print('Hello World!')
        issue_manager.datalab.find_issues(features=embeddings['embedding'], issue_types={'outlier': {'k': 3}})
        issue_manager.find_issues()
        (issues, summary, info) = (issue_manager.issues, issue_manager.summary, issue_manager.info)
        assert isinstance(issues, pd.DataFrame), 'Issues should be a dataframe'
        assert isinstance(summary, pd.DataFrame), 'Summary should be a dataframe'
        assert summary['issue_type'].values[0] == 'data_valuation'
        assert isinstance(info, dict), 'Info should be a dict'
        assert isinstance(issues, pd.DataFrame), 'Issues should be a dataframe'
        info_keys = info.keys()
        expected_keys = ['num_low_valuation_issues', 'average_data_valuation']
        assert all([key in info_keys for key in expected_keys]), f'Info should have the right keys, but is missing {set(expected_keys) - set(info_keys)}'

    def test_find_issue_wrong_knn_graph(self, issue_manager, embeddings):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(AssertionError):
            issue_manager.datalab.find_issues(features=embeddings['embedding'], issue_types={'outlier': {'k': 3}})
            issue_manager.find_issues(k=4)