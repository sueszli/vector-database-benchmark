from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['coolprop'])
def test_simple_propssi(selenium):
    if False:
        i = 10
        return i + 15
    from CoolProp.CoolProp import PropsSI
    assert round(PropsSI('T', 'P', 101325, 'Q', 0, 'Water'), 3) == 373.124

@run_in_pyodide(packages=['coolprop'])
def test_simple_phasesi(selenium):
    if False:
        print('Hello World!')
    from CoolProp.CoolProp import PhaseSI
    assert PhaseSI('P', 101325, 'Q', 0, 'Water') == 'twophase'