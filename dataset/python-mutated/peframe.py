from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class PEframe(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'PEframe'
    url: str = 'http://malware_tools_analyzers:4002/peframe'
    max_tries: int = 25
    poll_distance: int = 5
    timeout: int = 60 * 9

    def run(self):
        if False:
            while True:
                i = 10
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        req_data = {'args': ['-j', f'@{fname}'], 'timeout': self.timeout}
        req_files = {fname: binary}
        result = self._docker_run(req_data, req_files)
        if result:
            if 'strings' in result and 'dump' in result['strings']:
                result['strings']['dump'] = result['strings']['dump'][:100]
        return result