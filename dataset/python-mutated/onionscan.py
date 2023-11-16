from api_app.analyzers_manager.classes import DockerBasedAnalyzer, ObservableAnalyzer

class Onionscan(ObservableAnalyzer, DockerBasedAnalyzer):
    name: str = 'Onionscan'
    url: str = 'http://tor_analyzers:4001/onionscan'
    max_tries: int = 60
    poll_distance: int = 10
    verbose: bool
    tor_proxy_address: str

    def run(self):
        if False:
            return 10
        args = []
        if self.verbose:
            args.append('-verbose')
        if self.tor_proxy_address:
            args.extend(['-torProxyAddress', self.tor_proxy_address])
        args.extend(['-jsonReport', self.observable_name])
        req_data = {'args': args}
        return self._docker_run(req_data=req_data, req_files=None)