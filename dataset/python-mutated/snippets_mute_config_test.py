import os
import re
import uuid
from _pytest.capture import CaptureFixture
import backoff
from google.api_core.exceptions import InternalServerError, NotFound, ServiceUnavailable
from google.cloud import securitycenter
from google.cloud.securitycenter_v1.services.security_center.pagers import ListFindingsPager
import pytest
import snippets_mute_config
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
ORGANIZATION_ID = os.environ['GCLOUD_ORGANIZATION']
GOOGLE_APPLICATION_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

@pytest.fixture
def mute_rule():
    if False:
        return 10
    mute_rule_create = f'random-mute-create-{uuid.uuid4()}'
    mute_rule_update = f'random-mute-update-{uuid.uuid4()}'
    snippets_mute_config.create_mute_rule(f'projects/{PROJECT_ID}', mute_rule_create)
    snippets_mute_config.create_mute_rule(f'projects/{PROJECT_ID}', mute_rule_update)
    yield {'create': mute_rule_create, 'update': mute_rule_update}
    snippets_mute_config.delete_mute_rule(f'projects/{PROJECT_ID}/muteConfigs/{mute_rule_create}')
    snippets_mute_config.delete_mute_rule(f'projects/{PROJECT_ID}/muteConfigs/{mute_rule_update}')

@pytest.fixture
def finding(capsys: CaptureFixture):
    if False:
        i = 10
        return i + 15
    import snippets_findings
    from snippets_findings import create_finding
    snippets_findings.create_source(ORGANIZATION_ID)
    (out, _) = capsys.readouterr()
    source_path = out.split(':')[1].strip()
    source_name = source_path.split('/')[3]
    finding1_path = create_finding(source_path, '1testingscc').name
    finding2_path = create_finding(source_path, '2testingscc').name
    yield {'source': source_name, 'finding1': finding1_path, 'finding2': finding2_path}

def list_all_findings(source_name) -> ListFindingsPager:
    if False:
        i = 10
        return i + 15
    client = securitycenter.SecurityCenterClient()
    return client.list_findings(request={'parent': source_name})

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_get_mute_rule(capsys: CaptureFixture, mute_rule):
    if False:
        while True:
            i = 10
    snippets_mute_config.get_mute_rule(f"projects/{PROJECT_ID}/muteConfigs/{mute_rule.get('create')}")
    (out, _) = capsys.readouterr()
    assert re.search('Retrieved the mute rule: ', out)
    assert re.search(mute_rule.get('create'), out)

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_list_mute_rules(capsys: CaptureFixture, mute_rule):
    if False:
        i = 10
        return i + 15
    snippets_mute_config.list_mute_rules(f'projects/{PROJECT_ID}')
    (out, _) = capsys.readouterr()
    assert re.search(mute_rule.get('create'), out)
    assert re.search(mute_rule.get('update'), out)

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_update_mute_rule(capsys: CaptureFixture, mute_rule):
    if False:
        return 10
    snippets_mute_config.update_mute_rule(f"projects/{PROJECT_ID}/muteConfigs/{mute_rule.get('update')}")
    snippets_mute_config.get_mute_rule(f"projects/{PROJECT_ID}/muteConfigs/{mute_rule.get('update')}")
    (out, _) = capsys.readouterr()
    assert re.search('Updated mute config description', out)

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_set_mute_finding(capsys: CaptureFixture, finding):
    if False:
        i = 10
        return i + 15
    finding_path = finding.get('finding1')
    snippets_mute_config.set_mute_finding(finding_path)
    (out, _) = capsys.readouterr()
    assert re.search('Mute value for the finding: MUTED', out)

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_set_unmute_finding(capsys: CaptureFixture, finding):
    if False:
        for i in range(10):
            print('nop')
    finding_path = finding.get('finding1')
    snippets_mute_config.set_unmute_finding(finding_path)
    (out, _) = capsys.readouterr()
    assert re.search('Mute value for the finding: UNMUTED', out)

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, NotFound), max_tries=3)
def test_bulk_mute_findings(capsys: CaptureFixture, finding):
    if False:
        return 10
    snippets_mute_config.bulk_mute_findings(f'projects/{PROJECT_ID}', f'resource.project_display_name="{PROJECT_ID}"')
    response = list_all_findings(f"projects/{PROJECT_ID}/sources/{finding.get('source')}")
    for (i, finding) in enumerate(response):
        assert finding.finding.mute == securitycenter.Finding.Mute.MUTED