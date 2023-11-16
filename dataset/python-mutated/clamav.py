import logging
from api_app.analyzers_manager.classes import DockerBasedAnalyzer, FileAnalyzer
from tests.mock_utils import MockUpResponse
logger = logging.getLogger(__name__)

class ClamAV(FileAnalyzer, DockerBasedAnalyzer):
    name: str = 'ClamAV'
    url: str = 'http://malware_tools_analyzers:4002/clamav'
    poll_distance: int = 3
    max_tries: int = 20
    timeout: int = 60

    def run(self):
        if False:
            i = 10
            return i + 15
        binary = self.read_file_bytes()
        fname = str(self.filename).replace('/', '_').replace(' ', '_')
        args = [f'@{fname}']
        req_data = {'args': args, 'timeout': self.timeout}
        req_files = {fname: binary}
        report = self._docker_run(req_data, req_files)
        detections = []
        if 'Infected files: 1' in report:
            lines = report.split('\n')
            for line in lines:
                if 'SUMMARY' in line:
                    break
                words = line.split()
                if words:
                    signature = words[1]
                    logger.info(f'extracted signature {signature} for {self.job_id}')
                    detections.append(signature)
            if not detections:
                logger.error(f'no detections extracted? {self.job_id}')
        return {'detections': list(set(detections)), 'raw_report': report}

    @staticmethod
    def mocked_docker_analyzer_get(*args, **kwargs):
        if False:
            print('Hello World!')
        return MockUpResponse({'key': 'test', 'returncode': 0, 'report': 'OK real_signature\n'}, 200)