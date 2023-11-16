import pytest
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.data_issues import DataIssues

class TestDataIssues:
    labels = ['B', 'A', 'B']
    label_name = 'labels'

    @pytest.fixture
    def data_issues(self):
        if False:
            i = 10
            return i + 15
        data = Data(data={self.label_name: self.labels}, label_name=self.label_name)
        data_issues = DataIssues(data=data)
        yield data_issues

    def test_data_issues_init(self, data_issues):
        if False:
            print('Hello World!')
        assert hasattr(data_issues, 'issues')
        assert hasattr(data_issues, 'issue_summary')
        assert hasattr(data_issues, 'info')

    def test_statistics(self, data_issues):
        if False:
            while True:
                i = 10
        stats = data_issues.statistics
        assert stats == data_issues.info['statistics']
        assert stats['num_examples'] == 3, f"Incorrect number of examples: {stats['num_examples']}"
        assert stats['class_names'] == ['A', 'B'], f"Incorrect class names: {stats['class_names']}"
        assert stats['num_classes'] == 2, f"Incorrect number of classes: {stats['num_classes']}"
        assert stats['multi_label'] is False
        assert stats['health_score'] is None, f"Health score should initially be None, but is {stats['health_score']}"

    def test_get_info(self, data_issues):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            data_issues.get_info('nonexistent_issue')
        assert data_issues.get_info('statistics') == data_issues.info['statistics']

    def test_get_info_label(self, data_issues):
        if False:
            return 10
        data_issues.info['label'] = {'given_label': [0, 1, 1], 'predicted_label': [1, 0, 1]}
        info = data_issues.get_info('label')
        label_format_error_message = "get_info('label') should return the given label formatted with the class names"
        assert info.get('given_label').tolist() == ['A', 'B', 'B'], label_format_error_message
        assert info.get('predicted_label').tolist() == self.labels, label_format_error_message