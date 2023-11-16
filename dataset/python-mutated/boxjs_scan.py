from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class BoxJS(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'box-js'
    url: str = 'http://malware_tools_analyzers:4002/boxjs'
    max_tries: int = 5
    poll_distance: int = 12

    def run(self):
        if False:
            print('Hello World!')
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        binary = self.read_file_bytes()
        args = [f'@{fname}', '--output-dir=/tmp/boxjs', '--no-kill', '--no-shell-error', '--no-echo']
        req_data = {'args': args, 'timeout': 10, 'callback_context': {'read_result_from': fname}}
        req_files = {fname: binary}
        return self._docker_run(req_data, req_files)