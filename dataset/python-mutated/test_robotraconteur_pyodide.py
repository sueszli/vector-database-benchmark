from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['RobotRaconteur', 'numpy'])
def test_robotraconteur_import(selenium):
    if False:
        i = 10
        return i + 15
    from RobotRaconteur.Client import RRN
    _ = RRN.RobotRaconteurVersion

@run_in_pyodide(packages=['RobotRaconteur', 'numpy'])
def test_robotraconteur_exceptions(selenium):
    if False:
        while True:
            i = 10
    import pytest
    import RobotRaconteur as RR
    RRN = RR.RobotRaconteurNode.s
    RRN.SetNodeName('test_node')
    assert RRN.NodeName == 'test_node'
    with pytest.raises(Exception):
        RRN.SetNodeName('test_node')