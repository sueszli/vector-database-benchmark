import json
import os
import subprocess
import sys
import tempfile
import unittest

class TestPlannerReLaunch(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def test_relaunch_with_planner(self):
        if False:
            print('Hello World!')
        from test_auto_parallel_relaunch import cluster_json, mapping_josn
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        mapping_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_rank_mapping.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        mapping_json_object = json.loads(mapping_josn)
        with open(mapping_json_path, 'w') as mapping_json_file:
            json.dump(mapping_json_object, mapping_json_file)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(file_dir, 'auto_parallel_relaunch_with_planner.py')
        if os.environ.get('WITH_COVERAGE', 'OFF') == 'ON':
            coverage_args = ['-m', 'coverage', 'run', '--branch', '-p']
        else:
            coverage_args = []
        cmd = [sys.executable, '-u'] + coverage_args + ['-m', 'paddle.distributed.launch', '--log_dir', self.temp_dir.name, '--cluster_topo_path', cluster_json_path, '--rank_mapping_path', mapping_json_path, '--enable_auto_mapping', 'True', launch_model_path]
        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)
if __name__ == '__main__':
    unittest.main()