import pytest
from pytest_pyodide import run_in_pyodide

@pytest.mark.driver_timeout(40)
@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_version(selenium):
    if False:
        print('Hello World!')
    import sisl
    assert sisl.__version__ == '0.14.2'

@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_nodes(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    pytest.main(['--pyargs', 'sisl.nodes'])

@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_geom(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    pytest.main(['--pyargs', 'sisl.geom'])

@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_linalg(selenium):
    if False:
        while True:
            i = 10
    import pytest
    pytest.main(['--pyargs', 'sisl.linalg'])

@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_sparse(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    pytest.main(['--pyargs', 'sisl.tests.test_sparse'])

@run_in_pyodide(packages=['sisl-tests', 'pytest'])
def test_physics_sparse(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    pytest.main(['--pyargs', 'sisl.physics.tests.test_physics_sparse'])