from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['pyyaml'])
def test_pyyaml(selenium):
    if False:
        while True:
            i = 10
    import yaml
    from yaml import CLoader as Loader
    document = '\n    - Hesperiidae\n    - Papilionidae\n    - Apatelodidae\n    - Epiplemidae\n    '
    loaded = yaml.load(document, Loader=Loader)
    assert loaded == ['Hesperiidae', 'Papilionidae', 'Apatelodidae', 'Epiplemidae']