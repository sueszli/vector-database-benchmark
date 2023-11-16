import pytest
from marshmallow import fields
from marshmallow.schema import Schema
from integrations.lead_tracking.pipedrive.models import BasePipedriveModel, PipedriveOrganization

def test_base_pipedrive_model_from_api_response():
    if False:
        for i in range(10):
            print('nop')

    class MySchema(Schema):
        foo = fields.Str()

    class MyClass(BasePipedriveModel):
        schema = MySchema()

        def __init__(self, foo: str):
            if False:
                i = 10
                return i + 15
            self.foo = foo
    o = MyClass.from_response_data(data={'foo': 'bar'})
    assert o.foo == 'bar'

@pytest.mark.parametrize('domain, expected_org_name', (('example.com', 'example'), ('foo.example.com', 'example')))
def test_pipedrive_organisation_get_org_name_from_domain(domain, expected_org_name):
    if False:
        for i in range(10):
            print('nop')
    assert PipedriveOrganization.get_org_name_from_domain(domain) == expected_org_name

def test_pipedrive_organisation_get_org_name_from_domain_raises_value_error_if_domain_invalid():
    if False:
        return 10
    with pytest.raises(ValueError):
        PipedriveOrganization.get_org_name_from_domain('not_a_domain')