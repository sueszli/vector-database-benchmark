from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class CapaInfo(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'Capa'
    url: str = 'http://malware_tools_analyzers:4002/capa'
    poll_distance: int = 10
    max_tries: int = 60
    timeout: int = 60 * 9
    shellcode: bool
    arch: str

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        self.args = []
        if self.arch != '64':
            self.arch = '32'
        if self.shellcode:
            self.args.append('-f')
            self.args.append('sc' + self.arch)

    def run(self):
        if False:
            while True:
                i = 10
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        args = [f'@{fname}', *self.args]
        req_data = {'args': args, 'timeout': self.timeout}
        req_files = {fname: binary}
        return self._docker_run(req_data, req_files)