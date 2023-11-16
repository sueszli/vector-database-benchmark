from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class APKiD(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'apk_analyzers'
    url: str = 'http://malware_tools_analyzers:4002/apkid'
    max_tries: int = 10
    poll_distance: int = 3

    def run(self):
        if False:
            i = 10
            return i + 15
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        binary = self.read_file_bytes()
        args = ['-t', '20', '-j', f'@{fname}']
        req_data = {'args': args}
        req_files = {fname: binary}
        report = self._docker_run(req_data, req_files, analyzer_name=self.analyzer_name)
        if not report:
            self.report.errors.append('APKiD does not support the file')
            return {}
        return report