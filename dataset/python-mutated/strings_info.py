from json import dumps as json_dumps
from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class StringsInfo(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'StringsInfo'
    url: str = 'http://malware_tools_analyzers:4002/stringsifter'
    poll_distance: int = 10
    max_tries: int = 60
    timeout: int = 60 * 9
    max_number_of_strings: int
    max_characters_for_string: int
    rank_strings: int

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        args = ['flarestrings', f'@{fname}']
        req_data = {'args': args, 'timeout': self.timeout}
        req_files = {fname: binary}
        result = self._docker_run(req_data, req_files)
        exceed_max_strings = len(result) > self.max_number_of_strings
        if exceed_max_strings:
            result = list(result[:self.max_number_of_strings])
        if self.rank_strings:
            args = ['rank_strings', '--limit', str(self.max_number_of_strings), '--strings', json_dumps(result)]
            req_data = {'args': args, 'timeout': self.timeout}
            result = self._docker_run(req_data)
        result = {'data': [row[:self.max_characters_for_string] for row in result], 'exceeded_max_number_of_strings': exceed_max_strings}
        return result