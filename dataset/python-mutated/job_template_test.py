import os
import uuid
from google.api_core.exceptions import NotFound
import pytest
import create_job_template
import delete_job_template
import get_job_template
import list_job_templates
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
project_number = os.environ['GOOGLE_CLOUD_PROJECT_NUMBER']
template_id = f'my-python-test-template-{uuid.uuid4()}'

def test_template_operations(capsys: pytest.fixture) -> None:
    if False:
        return 10
    job_template_name = f'projects/{project_number}/locations/{location}/jobTemplates/{template_id}'
    try:
        delete_job_template.delete_job_template(project_id, location, template_id)
    except NotFound as e:
        print(f'Ignoring NotFound, details: {e}')
    (out, _) = capsys.readouterr()
    response = create_job_template.create_job_template(project_id, location, template_id)
    assert job_template_name in response.name
    response = get_job_template.get_job_template(project_id, location, template_id)
    assert job_template_name in response.name
    list_job_templates.list_job_templates(project_id, location)
    (out, _) = capsys.readouterr()
    assert job_template_name in out
    response = delete_job_template.delete_job_template(project_id, location, template_id)
    assert response is None