import quickstart_analyze_org_policy_governed_containers
ORG_ID = '474566717491'
CONSTRAINT_NAME = 'constraints/compute.requireOsLogin'

def test_analyze_org_policy_governed_containers(capsys):
    if False:
        for i in range(10):
            print('nop')
    quickstart_analyze_org_policy_governed_containers.analyze_org_policy_governed_containers(ORG_ID, CONSTRAINT_NAME)
    (out, _) = capsys.readouterr()
    assert CONSTRAINT_NAME in out