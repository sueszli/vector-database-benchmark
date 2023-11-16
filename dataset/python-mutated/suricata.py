from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer

class Suricata(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'Suricata'
    url: str = 'http://pcap_analyzers:4004/suricata'
    poll_distance: int = 3
    max_tries: int = 100
    timeout: int = 200
    reload_rules: bool
    extended_logs: bool

    def config(self):
        if False:
            return 10
        super().config()
        self.signatures = {}

    def run(self):
        if False:
            while True:
                i = 10
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        args = [f'@{fname}', self.md5, str(self.job_id)]
        if self.reload_rules:
            args.append('--reload_rules')
        req_data = {'args': args, 'timeout': self.timeout, 'callback_context': {'read_result_from': f'/tmp/eve_{self.job_id}'}}
        req_files = {fname: binary}
        report = self._docker_run(req_data, req_files)
        for detection in report.get('data', []):
            alert = detection.get('alert', {})
            signature_name = alert.get('signature')
            if signature_name not in self.signatures:
                self.signatures[signature_name] = {'alerts_triggered': 0, 'protocols': [], 'src_ips': [], 'dest_ips': [], 'src_ports': [], 'dest_ports': []}
                self.signatures[signature_name].update(alert)
                if 'signature' in self.signatures[signature_name]:
                    del self.signatures[signature_name]['signature']
            proto = detection.get('proto', '')
            src_ip = detection.get('src_ip', '')
            dest_ip = detection.get('dest_ip', '')
            src_port = detection.get('src_port', '')
            if src_port >= 49152:
                src_port = '>49152'
            dest_port = detection.get('dest_port', '')
            if dest_port >= 49152:
                dest_port = '>49152'
            self.signatures[signature_name]['alerts_triggered'] += 1
            self._add_item_to_signatures(proto, 'protocols', signature_name)
            self._add_item_to_signatures(src_ip, 'src_ips', signature_name)
            self._add_item_to_signatures(dest_ip, 'dest_ips', signature_name)
            self._add_item_to_signatures(src_port, 'src_ports', signature_name)
            self._add_item_to_signatures(dest_port, 'dest_ports', signature_name)
        report['signatures'] = self.signatures
        if not self.extended_logs and 'data' in report:
            del report['data']
        return report

    def _add_item_to_signatures(self, item, key, signature_name):
        if False:
            while True:
                i = 10
        item = str(item)
        if item and item not in self.signatures[signature_name][key]:
            self.signatures[signature_name][key].append(item)