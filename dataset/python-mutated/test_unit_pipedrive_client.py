import json
from os.path import abspath, dirname, join
import responses
from integrations.lead_tracking.pipedrive.constants import MarketingStatus

@responses.activate
def test_pipedrive_api_client_create_lead(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        while True:
            i = 10
    example_response_file_path = join(dirname(abspath(__file__)), 'example_api_responses/create_lead.json')
    title = 'Johnny Bravo'
    organization_id = 1
    lead_id = '11c18740-659d-11ed-b6e9-ab3d83dc63a5'
    with open(example_response_file_path) as f:
        responses.add(method=responses.POST, url=f'{pipedrive_base_url}/leads', json=json.load(f), status=201)
    lead = pipedrive_api_client.create_lead(title=title, organization_id=organization_id)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    request_body = json.loads(call.request.body.decode('utf-8'))
    assert request_body == {'title': title, 'organization_id': organization_id, 'label_ids': []}
    assert call.request.params['api_token'] == pipedrive_api_token
    assert lead.id == lead_id
    assert lead.title == title
    assert lead.organization_id == organization_id

@responses.activate
def test_pipedrive_api_client_create_organization(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        while True:
            i = 10
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/create_organization.json')
    name = 'Test org'
    organization_id = 1
    organization_field_key = '1ebc98029a711f60a51b7169b5784fa85d83f4cc'
    organization_field_value = 'some-value'
    with open(example_response_file_name) as f:
        responses.add(method=responses.POST, url=f'{pipedrive_base_url}/organizations', json=json.load(f), status=201)
    organization = pipedrive_api_client.create_organization(name=name, organization_fields={organization_field_key: organization_field_value})
    assert len(responses.calls) == 1
    call = responses.calls[0]
    request_body = json.loads(call.request.body.decode('utf-8'))
    assert request_body == {'name': name, organization_field_key: organization_field_value}
    assert call.request.params['api_token'] == pipedrive_api_token
    assert organization.id == organization_id
    assert organization.name == name

@responses.activate
def test_pipedrive_api_client_search_organizations(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        i = 10
        return i + 15
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/search_organizations.json')
    search_term = 'Test org'
    result_organization_name = 'Test org'
    result_organization_id = 1
    with open(example_response_file_name) as f:
        responses.add(method=responses.GET, url=f'{pipedrive_base_url}/organizations/search', json=json.load(f), status=200)
    organizations = pipedrive_api_client.search_organizations(search_term=search_term)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    assert call.request.params['term'] == search_term
    assert call.request.params['fields'] == 'custom_fields'
    assert len(organizations) == 1
    assert organizations[0].name == result_organization_name
    assert organizations[0].id == result_organization_id

@responses.activate
def test_pipedrive_api_client_search_persons(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        return 10
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/search_persons.json')
    search_term = 'johnnybravo@mailinator.com'
    result_person_name = 'Johnny Bravo'
    result_person_id = 1
    with open(example_response_file_name) as f:
        responses.add(method=responses.GET, url=f'{pipedrive_base_url}/persons/search', json=json.load(f), status=200)
    persons = pipedrive_api_client.search_persons(search_term=search_term)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    assert call.request.params['term'] == search_term
    assert len(persons) == 1
    assert persons[0].name == result_person_name
    assert persons[0].id == result_person_id

@responses.activate
def test_pipedrive_api_client_create_organization_field(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        return 10
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/create_organization_field.json')
    organization_field_name = 'new-field'
    organization_field_key = '1ebc98029a711f60a51b7169b5784fa85d83f4cc'
    with open(example_response_file_name) as f:
        responses.add(method=responses.POST, url=f'{pipedrive_base_url}/organizationFields', json=json.load(f), status=201)
    organization_field = pipedrive_api_client.create_organization_field(name=organization_field_name)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    assert organization_field.key == organization_field_key
    assert organization_field.name == organization_field_name

@responses.activate
def test_pipedrive_api_client_create_deal_field(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        for i in range(10):
            print('nop')
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/create_deal_field.json')
    deal_field_name = 'new-field'
    deal_field_key = '8a66c8cbf4295894315aef845661469fd98f0842'
    with open(example_response_file_name) as f:
        responses.add(method=responses.POST, url=f'{pipedrive_base_url}/dealFields', json=json.load(f), status=201)
    organization_field = pipedrive_api_client.create_deal_field(name=deal_field_name)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    assert organization_field.key == deal_field_key
    assert organization_field.name == deal_field_name

@responses.activate
def test_pipedrive_api_client_create_person(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        return 10
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/create_person.json')
    person_name = 'Yogi Bear'
    person_email = 'yogi.bear@testing.com'
    marketing_status = MarketingStatus.SUBSCRIBED
    person_id = 1
    with open(example_response_file_name) as f:
        responses.add(method=responses.POST, url=f'{pipedrive_base_url}/persons', json=json.load(f), status=201)
    person = pipedrive_api_client.create_person(name=person_name, email=person_email, marketing_status=marketing_status)
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    json_request_body = json.loads(call.request.body)
    assert json_request_body['name'] == person_name
    assert json_request_body['email'] == person_email
    assert json_request_body['marketing_status'] == marketing_status
    assert person.name == person_name
    assert person.id == person_id

@responses.activate
def test_pipedrive_api_client_list_lead_labels(pipedrive_api_client, pipedrive_base_url, pipedrive_api_token):
    if False:
        while True:
            i = 10
    example_response_file_name = join(dirname(abspath(__file__)), 'example_api_responses/list_lead_labels.json')
    result_label_id = 'f08b42a0-4e75-11ea-9643-03698ef1cfd6'
    result_label_name = 'Hot'
    with open(example_response_file_name) as f:
        responses.add(method=responses.GET, url=f'{pipedrive_base_url}/leadLabels', json=json.load(f), status=200)
    persons = pipedrive_api_client.list_lead_labels()
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call.request.params['api_token'] == pipedrive_api_token
    assert len(persons) == 1
    assert persons[0].name == result_label_name
    assert persons[0].id == result_label_id