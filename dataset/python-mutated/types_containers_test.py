from docker.types.containers import ContainerConfig

def test_uid_0_is_not_elided():
    if False:
        i = 10
        return i + 15
    x = ContainerConfig(image='i', version='v', command='true', user=0)
    assert x['User'] == '0'