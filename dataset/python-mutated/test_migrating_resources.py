from docs_snippets.guides.dagster.migrating_to_python_resources_and_config.migrating_resources import convert_resource, initial_code_base, new_resource_code_contextmanager, new_style_resource_on_context, new_style_resource_on_param, new_third_party_resource_fixed, new_third_party_resource_old_code_broken, old_resource_code_contextmanager, old_third_party_resource

def test_initial_code_base() -> None:
    if False:
        while True:
            i = 10
    defs = initial_code_base()
    assert defs.get_implicit_global_asset_job_def().execute_in_process().success

def test_convert_resource() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert convert_resource

def test_new_style_resource_on_context() -> None:
    if False:
        while True:
            i = 10
    assert new_style_resource_on_context

def test_new_style_resource_on_param() -> None:
    if False:
        while True:
            i = 10
    assert new_style_resource_on_param

def test_old_third_party_resource() -> None:
    if False:
        i = 10
        return i + 15
    defs = old_third_party_resource()
    assert defs.get_implicit_global_asset_job_def().execute_in_process().success

def test_old_resource_code_contextmanager() -> None:
    if False:
        while True:
            i = 10
    defs = old_resource_code_contextmanager()
    assert defs.get_implicit_global_asset_job_def().execute_in_process().success

def test_new_resource_code_contextmanager() -> None:
    if False:
        i = 10
        return i + 15
    defs = new_resource_code_contextmanager()
    assert defs.get_implicit_global_asset_job_def().execute_in_process().success

def test_new_third_party_resource_old_code_broken() -> None:
    if False:
        while True:
            i = 10
    defs = new_third_party_resource_old_code_broken()
    assert defs.get_job_def('new_asset_job').execute_in_process().success
    assert not defs.get_job_def('existing_asset_job').execute_in_process(raise_on_error=False).success

def test_new_third_party_resource_fixed() -> None:
    if False:
        return 10
    defs = new_third_party_resource_fixed()
    assert defs.get_job_def('new_asset_job').execute_in_process().success
    assert defs.get_job_def('existing_asset_job').execute_in_process().success