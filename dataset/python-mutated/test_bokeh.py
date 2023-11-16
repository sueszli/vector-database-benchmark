from pytest_pyodide.decorator import run_in_pyodide

@run_in_pyodide(packages=['bokeh'])
def test_bokeh(selenium):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check whether any errors occur when drawing a basic plot.\n    Intended to function as a regression test.\n    '
    from bokeh.plotting import figure
    fig = figure()
    fig.line(range(3), [1, 4, 6])
    del fig