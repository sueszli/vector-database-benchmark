from e2b.templates.data_analysis import DataAnalysis

def test_create_graph():
    if False:
        i = 10
        return i + 15
    s = DataAnalysis()
    (a, b, artifacts) = s.run_python("\nimport matplotlib.pyplot as plt\n\nplt.plot([1, 2, 3, 4])\nplt.ylabel('some numbers')\nplt.show()\n    ")
    s.close()
    assert len(artifacts) == 1

def test_install_packages():
    if False:
        return 10
    s = DataAnalysis()
    s.install_python_packages('pandas')
    s.install_python_packages(['pandas'])
    s.install_python_packages(' ')
    s.install_python_packages([])
    s.install_system_packages('curl')
    s.install_system_packages(['curl'])
    s.install_system_packages('')
    s.install_system_packages([])
    s.close()