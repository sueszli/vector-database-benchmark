import sys
sys.path.insert(1, '../../')
import h2o
import zipfile
import os
from tests import pyunit_utils

def find_marker(path, marker):
    if False:
        return 10
    for line in open(path, 'r'):
        if marker in line:
            return True
    return False

def test_download_txt_logs():
    if False:
        return 10
    marker = 'txt-logs-marker'
    results_dir = pyunit_utils.locate('results')
    h2o.log_and_echo(marker)
    log_path = h2o.download_all_logs(results_dir, filename='plain_text_logs.log_ignore', container='LOG')
    assert not zipfile.is_zipfile(log_path)
    assert find_marker(log_path, marker) or os.path.getsize(log_path) > 10 * 1024 * 1042
    zip_path = h2o.download_all_logs(results_dir, filename='zip_logs.zip')
    assert zipfile.is_zipfile(zip_path)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_download_txt_logs)
else:
    test_download_txt_logs()