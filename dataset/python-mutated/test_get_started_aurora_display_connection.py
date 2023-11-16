import pytest
from scenario_get_started_aurora import AuroraClusterScenario

def test_display_connection(capsys):
    if False:
        i = 10
        return i + 15
    cluster = {'Endpoint': 'test-endpoint', 'Port': 1313, 'MasterUsername': 'test-user'}
    AuroraClusterScenario.display_connection(cluster)
    capt = capsys.readouterr()
    assert cluster['Endpoint'] in capt.out
    assert str(cluster['Port']) in capt.out
    assert cluster['MasterUsername'] in capt.out