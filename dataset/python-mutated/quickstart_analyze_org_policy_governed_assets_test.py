import quickstart_analyze_org_policy_governed_assets
ORG_ID = '474566717491'
CONSTRAINT_NAME = 'constraints/compute.requireOsLogin'

def test_analyze_org_policy_governed_assets(capsys):
    if False:
        while True:
            i = 10
    quickstart_analyze_org_policy_governed_assets.analyze_org_policy_governed_assets(ORG_ID, CONSTRAINT_NAME)
    (out, _) = capsys.readouterr()
    assert CONSTRAINT_NAME in out