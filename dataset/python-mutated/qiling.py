from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException

class Qiling(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'Qiling'
    url: str = 'http://malware_tools_analyzers:4002/qiling'
    max_tries: int = 15
    poll_distance: int = 30
    timeout: int = 60 * 9
    os: str
    arch: str
    shellcode: bool
    profile: str

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        super().config()
        self.args = [self.os, self.arch]
        if self.shellcode:
            self.args.append('--shellcode')
        if self.profile:
            self.args.extend(['--profile'] + [self.profile])

    def run(self):
        if False:
            i = 10
            return i + 15
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        binary = self.read_file_bytes()
        req_data = {'args': [f'@{fname}', *self.args], 'timeout': self.timeout}
        req_files = {fname: binary}
        report = self._docker_run(req_data, req_files)
        if report.get('setup_error'):
            raise AnalyzerRunException(report['setup_error'])
        if report.get('execution_error'):
            raise AnalyzerRunException(report['execution_error'])
        return report