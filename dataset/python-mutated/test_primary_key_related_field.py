import pytest
from rest_framework.exceptions import ValidationError
from awx.api.serializers import JobLaunchSerializer

@pytest.mark.parametrize('param', ['credentials', 'instance_groups', 'labels'])
def test_primary_key_related_field(param):
    if False:
        for i in range(10):
            print('nop')
    data = {param: {'1': '2', '3': '4'}}
    with pytest.raises(ValidationError):
        JobLaunchSerializer(data=data)