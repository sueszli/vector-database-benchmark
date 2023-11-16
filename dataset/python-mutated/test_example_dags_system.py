from __future__ import annotations
import pytest
from tests.test_utils.system_tests_class import SystemTest

@pytest.mark.system('core')
class TestExampleDagsSystem(SystemTest):

    @pytest.mark.parametrize('dag_id', ['example_bash_operator', 'example_branch_operator', 'tutorial_dag', 'example_dag_decorator'])
    def test_dag_example(self, dag_id):
        if False:
            for i in range(10):
                print('nop')
        self.run_dag(dag_id=dag_id)