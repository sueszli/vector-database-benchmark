from json import dumps as json_dumps
from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class Floss(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'Floss'
    url: str = 'http://malware_tools_analyzers:4002/floss'
    ranking_url: str = 'http://malware_tools_analyzers:4002/stringsifter'
    poll_distance: int = 10
    max_tries: int = 60
    timeout: int = 60 * 9
    OS_MAX_ARGS: int = 2097152
    max_no_of_strings: dict
    rank_strings: dict

    def run(self):
        if False:
            return 10
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        args = [f'@{fname}']
        req_data = {'args': args, 'timeout': self.timeout}
        req_files = {fname: binary}
        result = self._docker_run(req_data, req_files)
        result['exceeded_max_number_of_strings'] = {}
        self.url = self.ranking_url
        for key in self.max_no_of_strings:
            if self.rank_strings[key]:
                strings = json_dumps(result['strings'][key])
                analyzable_strings = strings[:self.OS_MAX_ARGS - 5]
                args = ['rank_strings', '--limit', str(self.max_no_of_strings[key]), '--strings', analyzable_strings]
                req_data = {'args': args, 'timeout': self.timeout}
                result['strings'][key] = self._docker_run(req_data)
            elif len(result.get('strings', {}).get(key, [])) > self.max_no_of_strings[key]:
                result['strings'][key] = list(result['strings'][key])
                result['exceeded_max_number_of_strings'][key] = True
        return result