import os
import subprocess
from lightning import BuildConfig, LightningWork

class Locust(LightningWork):

    def __init__(self, num_users: int=100):
        if False:
            while True:
                i = 10
        'This component checks the performance of a server. The server url is passed to its run method.\n\n        Arguments:\n            num_users: Number of users emulated by Locust\n\n        '
        super().__init__(port=8089, parallel=True, cloud_build_config=BuildConfig(requirements=['locust']))
        self.num_users = num_users

    def run(self, load_tested_url: str):
        if False:
            while True:
                i = 10
        cmd = ' '.join(['locust', '--master-host', str(self.host), '--master-port', str(self.port), '--host', str(load_tested_url), '-u', str(self.num_users)])
        process = subprocess.Popen(cmd, cwd=os.path.dirname(__file__), shell=True)
        process.wait()