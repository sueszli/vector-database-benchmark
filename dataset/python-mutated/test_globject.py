from vispy.testing import run_tests_if_main
from vispy.gloo.globject import GLObject

def test_globject():
    if False:
        return 10
    'Test gl object uinique id and GLIR CREATE command'
    objects = [GLObject() for i in range(10)]
    ids = [ob.id for ob in objects]
    assert len(set(ids)) == len(objects)
    commands = []
    for ob in objects:
        commands.extend(ob._glir.clear())
    assert len(commands) == len(objects)
    for cmd in commands:
        assert cmd[0] == 'CREATE'
    ob = objects[-1]
    q = ob._glir
    ob.delete()
    cmd = q.clear()[-1]
    assert cmd[0] == 'DELETE'
run_tests_if_main()