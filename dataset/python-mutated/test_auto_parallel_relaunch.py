import json
import os
import subprocess
import sys
import tempfile
import unittest
cluster_json = '\n{\n  "machines": [\n    {\n      "hostname": "machine1",\n      "addr": "127.0.0.1",\n      "port": "768",\n      "devices": [\n        {\n          "global_id": 0,\n          "local_id": 0,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 1,\n          "local_id": 1,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 2,\n          "local_id": 0,\n          "type": "CPU",\n          "model": "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G",\n          "arch": "x86_64",\n          "vendor": "GenuineIntel",\n          "sp_gflops": 150,\n          "dp_gflops": 75,\n          "memory": "503"\n        }\n      ],\n      "links": [\n        {\n          "source_global_id": 0,\n          "target_global_id": 1,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 0,\n          "type": "PHB",\n          "bandwidth": 12\n        }\n      ]\n    }\n  ]\n}\n'
mapping_josn = '\n[\n  {\n    "hostname": "machine1",\n    "addr": "127.0.0.1",\n    "port": "768",\n    "ranks":\n      {\n        "0": [1],\n        "1": [0]\n      }\n  }\n]\n'

class TestAutoParallelReLaunch(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_relaunch(self):
        if False:
            while True:
                i = 10
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        mapping_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_rank_mapping.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        mapping_josn_object = json.loads(mapping_josn)
        with open(mapping_json_path, 'w') as mapping_josn_file:
            json.dump(mapping_josn_object, mapping_josn_file)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(file_dir, 'auto_parallel_relaunch_model.py')
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