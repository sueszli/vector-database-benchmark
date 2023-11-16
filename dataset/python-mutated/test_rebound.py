from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['rebound', 'numpy'])
def test_integrate(selenium):
    if False:
        return 10
    import numpy
    import rebound
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=0.001, a=1)
    sim.add(m=0.001, a=2)
    sim.integrate(10)
    assert numpy.fabs(sim.t - 10.0) < 1e-10, 'Orbit integration not working'
    return None