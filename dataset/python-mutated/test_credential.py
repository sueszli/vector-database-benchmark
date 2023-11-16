import pytest
from awx.main.models import Credential, CredentialType

@pytest.mark.django_db
def test_unique_hash_with_unicode():
    if False:
        for i in range(10):
            print('nop')
    ct = CredentialType.objects.create(name='Väult', kind='vault')
    cred = Credential.objects.create(name='Iñtërnâtiônàlizætiøn', credential_type=ct, inputs={'vault_id': '🐉🐉🐉'})
    assert cred.unique_hash(display=True) == 'Väult (id=🐉🐉🐉)'

def test_custom_cred_with_empty_encrypted_field():
    if False:
        print('Hello World!')
    ct = CredentialType(name='My Custom Cred', kind='custom', inputs={'fields': [{'id': 'some_field', 'label': 'My Field', 'secret': True}]})
    cred = Credential(id=4, name='Testing 1 2 3', credential_type=ct, inputs={})
    assert cred.encrypt_field('some_field', None) is None