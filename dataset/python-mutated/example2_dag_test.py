import internal_unit_testing

def test_dag_import():
    if False:
        return 10
    from . import example2_dag
    internal_unit_testing.assert_has_valid_dag(example2_dag)