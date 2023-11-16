import quickstart_analyze_org_policies
ORG_ID = '474566717491'
CONSTRAINT_NAME = 'constraints/compute.requireOsLogin'

def test_analyze_org_policies(capsys):
    if False:
        print('Hello World!')
    quickstart_analyze_org_policies.analyze_org_policies(ORG_ID, CONSTRAINT_NAME)
    (out, _) = capsys.readouterr()
    assert CONSTRAINT_NAME in out