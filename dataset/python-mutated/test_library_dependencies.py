def test_eel_functions_and_variables_exist():
    if False:
        i = 10
        return i + 15
    ' Test that the functions and variables that we use from Eel exist. '
    import eel
    assert hasattr(eel, 'init')
    assert callable(eel.init)
    assert hasattr(eel, 'expose')
    assert callable(eel.expose)
    assert hasattr(eel, 'start')
    assert callable(eel.start)
    from eel import chrome
    chrome_path = chrome.find_path()
    assert chrome_path is None or isinstance(chrome_path, str)

def test_pyinstaller_functions_and_variables_exist():
    if False:
        print('Hello World!')
    ' Test that the functions and variables that we use from PyInstaller exist. '
    import PyInstaller.__main__ as pyi_main
    assert hasattr(pyi_main, 'run')
    assert callable(pyi_main.run)
    import PyInstaller as pyi
    assert isinstance(pyi.__version__, str)