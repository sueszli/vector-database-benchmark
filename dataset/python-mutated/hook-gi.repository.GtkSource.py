def pre_safe_import_module(api):
    if False:
        for i in range(10):
            print('nop')
    api.add_runtime_module(api.module_name)