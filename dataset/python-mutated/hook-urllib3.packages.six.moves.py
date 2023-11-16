from PyInstaller import isolated

def pre_safe_import_module(api):
    if False:
        return 10

    @isolated.call
    def real_to_six_module_name():
        if False:
            while True:
                i = 10
        import urllib3.packages.six as six
        return {moved.mod: 'urllib3.packages.six.moves.' + moved.name for moved in six._moved_attributes if isinstance(moved, (six.MovedModule, six.MovedAttribute))}
    api.add_runtime_package(api.module_name)
    for (real_module_name, six_module_name) in real_to_six_module_name.items():
        api.add_alias_module(real_module_name, six_module_name)