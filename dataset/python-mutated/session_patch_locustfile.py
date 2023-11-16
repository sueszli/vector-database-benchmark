import locust
from locust.user import task
from archivist.archivist import Archivist

class ArchivistUser(locust.HttpUser):

    def on_start(self):
        if False:
            for i in range(10):
                print('nop')
        AUTH_TOKEN = None
        with open('auth.text') as f:
            AUTH_TOKEN = f.read()
        self.arch: Archivist = Archivist(url=self.host, auth=AUTH_TOKEN)
        self.arch._session = self.client

    @task
    def Create_assets(self):
        if False:
            i = 10
            return i + 15
        'User creates assets as fast as possible'
        while True:
            self.arch.assets.create(behaviours=['Builtin', 'RecordEvidence', 'Attachments'], attrs={'foo': 'bar'})