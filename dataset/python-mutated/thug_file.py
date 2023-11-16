import secrets
from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class ThugFile(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'Thug'
    url: str = 'http://malware_tools_analyzers:4002/thug'
    max_tries: int = 15
    poll_distance: int = 30
    user_agent: str
    dom_events: str
    use_proxy: bool
    proxy: str
    enable_awis: bool
    enable_image_processing_analysis: bool

    def _thug_args_builder(self):
        if False:
            for i in range(10):
                print('nop')
        user_agent = self.user_agent
        dom_events = self.dom_events
        use_proxy = self.use_proxy
        proxy = self.proxy
        enable_awis = self.enable_awis
        enable_img_proc = self.enable_image_processing_analysis
        args = ['-T', '300', '-u', str(user_agent)]
        if dom_events:
            args.extend(['-e', str(dom_events)])
        if use_proxy and proxy:
            args.extend(['-p', str(proxy)])
        if enable_awis:
            args.append('--awis')
        if enable_img_proc:
            args.append('--image-processing')
        return args

    def run(self):
        if False:
            print('Hello World!')
        args = self._thug_args_builder()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        tmp_dir = f'{fname}_{secrets.token_hex(4)}'
        binary = self.read_file_bytes()
        args.extend(['-n', '/home/thug/' + tmp_dir, '-l', f'@{fname}'])
        req_data = {'args': args, 'callback_context': {'read_result_from': tmp_dir}}
        req_files = {fname: binary}
        return self._docker_run(req_data, req_files)