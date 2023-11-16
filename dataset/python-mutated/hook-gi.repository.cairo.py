def pre_safe_import_module(api):
    if False:
        i = 10
        return i + 15
    api.add_runtime_module(api.module_name)