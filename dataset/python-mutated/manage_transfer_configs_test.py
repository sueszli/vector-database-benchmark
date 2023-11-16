from . import manage_transfer_configs

def test_list_configs(capsys, project_id, transfer_config_name):
    if False:
        i = 10
        return i + 15
    manage_transfer_configs.list_configs({'project_id': project_id})
    (out, _) = capsys.readouterr()
    assert 'Got the following configs:' in out
    assert transfer_config_name in out

def test_update_config(capsys, transfer_config_name):
    if False:
        while True:
            i = 10
    manage_transfer_configs.update_config({'new_display_name': 'name from test_update_config', 'transfer_config_name': transfer_config_name})
    (out, _) = capsys.readouterr()
    assert 'Updated config:' in out
    assert transfer_config_name in out
    assert 'name from test_update_config' in out

def test_update_credentials_with_service_account(capsys, project_id, service_account_name, transfer_config_name):
    if False:
        print('Hello World!')
    manage_transfer_configs.update_credentials_with_service_account({'project_id': project_id, 'service_account_name': service_account_name, 'transfer_config_name': transfer_config_name})
    (out, _) = capsys.readouterr()
    assert 'Updated config:' in out
    assert transfer_config_name in out

def test_schedule_backfill_manual_transfer(capsys, transfer_config_name):
    if False:
        return 10
    runs = manage_transfer_configs.schedule_backfill_manual_transfer({'transfer_config_name': transfer_config_name})
    (out, _) = capsys.readouterr()
    assert 'Started manual transfer runs:' in out
    assert transfer_config_name in out
    assert len(runs) == 3

def test_delete_config(capsys, transfer_config_name):
    if False:
        i = 10
        return i + 15
    assert len(transfer_config_name) != 0