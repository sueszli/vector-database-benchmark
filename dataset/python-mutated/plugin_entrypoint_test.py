from errbot.utils import entry_point_plugins

def test_entrypoint_paths():
    if False:
        i = 10
        return i + 15
    plugins = entry_point_plugins('console_scripts')
    match = False
    for plugin in plugins:
        if 'errbot/errbot.cli' in plugin:
            match = True
    assert match

def test_entrypoint_paths_empty():
    if False:
        while True:
            i = 10
    groups = ['errbot.plugins', 'errbot.backend_plugins']
    for entry_point_group in groups:
        plugins = entry_point_plugins(entry_point_group)
        assert plugins == []