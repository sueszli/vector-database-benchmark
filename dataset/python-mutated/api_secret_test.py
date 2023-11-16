import docker
import pytest
from ..helpers import force_leave_swarm, requires_api_version
from .base import BaseAPIIntegrationTest

@requires_api_version('1.25')
class SecretAPITest(BaseAPIIntegrationTest):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        client = cls.get_client_instance()
        force_leave_swarm(client)
        cls._init_swarm(client)

    @classmethod
    def teardown_class(cls):
        if False:
            i = 10
            return i + 15
        client = cls.get_client_instance()
        force_leave_swarm(client)

    def test_create_secret(self):
        if False:
            while True:
                i = 10
        secret_id = self.client.create_secret('favorite_character', 'sakuya izayoi')
        self.tmp_secrets.append(secret_id)
        assert 'ID' in secret_id
        data = self.client.inspect_secret(secret_id)
        assert data['Spec']['Name'] == 'favorite_character'

    def test_create_secret_unicode_data(self):
        if False:
            return 10
        secret_id = self.client.create_secret('favorite_character', 'いざよいさくや')
        self.tmp_secrets.append(secret_id)
        assert 'ID' in secret_id
        data = self.client.inspect_secret(secret_id)
        assert data['Spec']['Name'] == 'favorite_character'

    def test_inspect_secret(self):
        if False:
            return 10
        secret_name = 'favorite_character'
        secret_id = self.client.create_secret(secret_name, 'sakuya izayoi')
        self.tmp_secrets.append(secret_id)
        data = self.client.inspect_secret(secret_id)
        assert data['Spec']['Name'] == secret_name
        assert 'ID' in data
        assert 'Version' in data

    def test_remove_secret(self):
        if False:
            for i in range(10):
                print('nop')
        secret_name = 'favorite_character'
        secret_id = self.client.create_secret(secret_name, 'sakuya izayoi')
        self.tmp_secrets.append(secret_id)
        assert self.client.remove_secret(secret_id)
        with pytest.raises(docker.errors.NotFound):
            self.client.inspect_secret(secret_id)

    def test_list_secrets(self):
        if False:
            return 10
        secret_name = 'favorite_character'
        secret_id = self.client.create_secret(secret_name, 'sakuya izayoi')
        self.tmp_secrets.append(secret_id)
        data = self.client.secrets(filters={'names': ['favorite_character']})
        assert len(data) == 1
        assert data[0]['ID'] == secret_id['ID']