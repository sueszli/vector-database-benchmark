import pytest
from awx.api.serializers import OAuth2TokenSerializer

@pytest.mark.parametrize('scope, expect', [('', False), ('read', True), ('read read', False), ('write read', True), ('read rainbow', False)])
def test_invalid_scopes(scope, expect):
    if False:
        return 10
    assert OAuth2TokenSerializer()._is_valid_scope(scope) is expect