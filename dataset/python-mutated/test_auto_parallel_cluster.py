import json
import os
import tempfile
import unittest
from paddle.distributed.auto_parallel.static.cluster import Cluster, DeviceType, LinkType
cluster_json = '\n{\n  "machines": [\n    {\n      "hostname": "machine0",\n      "addr": "0.0.0.1",\n      "port": "768",\n      "devices": [\n        {\n          "global_id": 0,\n          "local_id": 0,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 1,\n          "local_id": 1,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 2,\n          "local_id": 0,\n          "type": "CPU",\n          "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",\n          "arch": "x86_64",\n          "vendor": "GenuineIntel",\n          "sp_gflops": 150,\n          "dp_gflops": 75,\n          "memory": 1510\n        },\n        {\n          "global_id": 3,\n          "local_id": 0,\n          "type": "NIC"\n        }\n      ],\n      "links": [\n        {\n          "source_global_id": 0,\n          "target_global_id": 1,\n          "type": "NVL",\n          "bandwidth": 252\n        },\n        {\n          "source_global_id": 0,\n          "target_global_id": 2,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 2,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 0,\n          "target_global_id": 3,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 3,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 2,\n          "target_global_id": 3,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 3,\n          "target_global_id": 7,\n          "type": "NET",\n          "bandwidth": 1\n        }\n      ]\n    },\n    {\n      "hostname": "machine1",\n      "addr": "0.0.0.2",\n      "port": "768",\n      "devices": [\n        {\n          "global_id": 4,\n          "local_id": 0,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 5,\n          "local_id": 1,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 6,\n          "local_id": 0,\n          "type": "CPU",\n          "model": "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G",\n          "arch": "x86_64",\n          "vendor": "GenuineIntel",\n          "sp_gflops": 150,\n          "dp_gflops": 75,\n          "memory": "503"\n        },\n        {\n          "global_id": 7,\n          "local_id": 0,\n          "type": "NIC"\n        }\n      ],\n      "links": [\n        {\n          "source_global_id": 4,\n          "target_global_id": 5,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 4,\n          "target_global_id": 6,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 5,\n          "target_global_id": 6,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 4,\n          "target_global_id": 7,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 5,\n          "target_global_id": 7,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 6,\n          "target_global_id": 7,\n          "type": "NET",\n          "bandwidth": 1\n        },\n        {\n          "source_global_id": 7,\n          "target_global_id": 3,\n          "type": "NET",\n          "bandwidth": 1\n        }\n      ]\n    }\n  ]\n}\n'

class TestAutoParallelCluster(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    def test_cluster(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        self.assertEqual(len(cluster.get_all_devices('GPU')), 4)
        self.assertEqual(len(cluster.get_all_devices('CPU')), 2)
        self.assertEqual(len(cluster.get_all_devices('NIC')), 2)
        self.assertEqual(len(cluster.machines), 2)
        machine0 = cluster.machines[0]
        self.assertEqual(machine0.id, 0)
        self.assertEqual(machine0.hostname, 'machine0')
        self.assertEqual(machine0.addr, '0.0.0.1')
        self.assertEqual(machine0.port, '768')
        self.assertEqual(len(machine0.devices), 4)
        self.assertEqual(len(machine0.links), 7)
        device0_machine0 = machine0.devices[0]
        self.assertEqual(device0_machine0.global_id, 0)
        self.assertEqual(device0_machine0.local_id, 0)
        self.assertEqual(device0_machine0.type, DeviceType.GPU)
        self.assertEqual(device0_machine0.model, 'A100-SXM4-40GB')
        self.assertAlmostEqual(device0_machine0.sp_gflops, 19500)
        self.assertAlmostEqual(device0_machine0.dp_gflops, 9700)
        self.assertAlmostEqual(device0_machine0.memory, 40)
        link0_machine0 = machine0.links[0, 1]
        self.assertEqual(link0_machine0.source.global_id, 0)
        self.assertEqual(link0_machine0.target.global_id, 1)
        self.assertEqual(link0_machine0.type, LinkType.NVL)
        self.assertAlmostEqual(link0_machine0.bandwidth, 252)
        self.assertAlmostEqual(link0_machine0.latency, 0)
        link1_machine0 = machine0.links[0, 2]
        self.assertEqual(link1_machine0.source.global_id, 0)
        self.assertEqual(link1_machine0.target.global_id, 2)
        self.assertEqual(link1_machine0.type, LinkType.PHB)
        self.assertAlmostEqual(link1_machine0.bandwidth, 12)
        self.assertAlmostEqual(link1_machine0.latency, 0)
        link2_machine0 = machine0.links[0, 3]
        self.assertEqual(link2_machine0.source.global_id, 0)
        self.assertEqual(link2_machine0.target.global_id, 3)
        self.assertEqual(link2_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link2_machine0.bandwidth, 1)
        self.assertAlmostEqual(link2_machine0.latency, 0)
        device1_machine0 = machine0.devices[1]
        self.assertEqual(device1_machine0.global_id, 1)
        self.assertEqual(device1_machine0.local_id, 1)
        self.assertEqual(device1_machine0.type, DeviceType.GPU)
        self.assertEqual(device1_machine0.model, 'A100-SXM4-40GB')
        self.assertAlmostEqual(device1_machine0.sp_gflops, 19500)
        self.assertAlmostEqual(device1_machine0.dp_gflops, 9700)
        self.assertAlmostEqual(device1_machine0.memory, 40)
        link0_machine0 = machine0.links[1, 2]
        self.assertEqual(link0_machine0.source.global_id, 1)
        self.assertEqual(link0_machine0.target.global_id, 2)
        self.assertEqual(link0_machine0.type, LinkType.PHB)
        self.assertAlmostEqual(link0_machine0.bandwidth, 12)
        self.assertAlmostEqual(link0_machine0.latency, 0)
        link1_machine0 = machine0.links[1, 3]
        self.assertEqual(link1_machine0.source.global_id, 1)
        self.assertEqual(link1_machine0.target.global_id, 3)
        self.assertEqual(link1_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link1_machine0.bandwidth, 1)
        self.assertAlmostEqual(link1_machine0.latency, 0)
        device2_machine0 = machine0.devices[2]
        self.assertEqual(device2_machine0.global_id, 2)
        self.assertEqual(device2_machine0.local_id, 0)
        self.assertEqual(device2_machine0.type, DeviceType.CPU)
        self.assertEqual(device2_machine0.model, 'Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH')
        self.assertAlmostEqual(device2_machine0.sp_gflops, 150)
        self.assertAlmostEqual(device2_machine0.dp_gflops, 75)
        self.assertAlmostEqual(device2_machine0.memory, 1510)
        link0_machine0 = machine0.links[2, 3]
        self.assertEqual(link0_machine0.source.global_id, 2)
        self.assertEqual(link0_machine0.target.global_id, 3)
        self.assertEqual(link0_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine0.bandwidth, 1)
        self.assertAlmostEqual(link0_machine0.latency, 0)
        device3_machine0 = machine0.devices[3]
        self.assertEqual(device3_machine0.global_id, 3)
        self.assertEqual(device3_machine0.local_id, 0)
        self.assertEqual(device3_machine0.type, DeviceType.NIC)
        self.assertAlmostEqual(device3_machine0.model, None)
        self.assertAlmostEqual(device3_machine0.sp_gflops, 0)
        self.assertAlmostEqual(device3_machine0.dp_gflops, 0)
        self.assertAlmostEqual(device3_machine0.memory, 0)
        link0_machine0 = machine0.links[3, 7]
        self.assertEqual(link0_machine0.source.global_id, 3)
        self.assertEqual(link0_machine0.target.global_id, 7)
        self.assertEqual(link0_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine0.bandwidth, 1)
        self.assertAlmostEqual(link0_machine0.latency, 0)
        machine1 = cluster.machines[1]
        self.assertEqual(machine1.id, 1)
        self.assertEqual(machine1.hostname, 'machine1')
        self.assertEqual(machine1.addr, '0.0.0.2')
        self.assertEqual(machine1.port, '768')
        self.assertEqual(len(machine1.devices), 4)
        self.assertEqual(len(machine1.links), 7)
        device4_machine1 = machine1.devices[4]
        self.assertEqual(device4_machine1.global_id, 4)
        self.assertEqual(device4_machine1.local_id, 0)
        self.assertEqual(device4_machine1.type, DeviceType.GPU)
        self.assertEqual(device4_machine1.model, 'Tesla V100-SXM2-32GB')
        self.assertAlmostEqual(device4_machine1.sp_gflops, 15700)
        self.assertAlmostEqual(device4_machine1.dp_gflops, 7800)
        self.assertAlmostEqual(device4_machine1.memory, 32)
        link0_machine1 = machine1.links[4, 5]
        self.assertEqual(link0_machine1.source.global_id, 4)
        self.assertEqual(link0_machine1.target.global_id, 5)
        self.assertEqual(link0_machine1.type, LinkType.NVL)
        self.assertAlmostEqual(link0_machine1.bandwidth, 42)
        self.assertAlmostEqual(link0_machine1.latency, 0)
        link1_machine1 = machine1.links[4, 6]
        self.assertEqual(link1_machine1.source.global_id, 4)
        self.assertEqual(link1_machine1.target.global_id, 6)
        self.assertEqual(link1_machine1.type, LinkType.PHB)
        self.assertAlmostEqual(link1_machine1.bandwidth, 12)
        self.assertAlmostEqual(link1_machine1.latency, 0)
        link2_machine1 = machine1.links[4, 7]
        self.assertEqual(link2_machine1.source.global_id, 4)
        self.assertEqual(link2_machine1.target.global_id, 7)
        self.assertEqual(link2_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link2_machine1.bandwidth, 1)
        self.assertAlmostEqual(link2_machine1.latency, 0)
        device5_machine1 = machine1.devices[5]
        self.assertEqual(device5_machine1.global_id, 5)
        self.assertEqual(device5_machine1.local_id, 1)
        self.assertEqual(device5_machine1.type, DeviceType.GPU)
        self.assertEqual(device4_machine1.model, 'Tesla V100-SXM2-32GB')
        self.assertAlmostEqual(device4_machine1.sp_gflops, 15700)
        self.assertAlmostEqual(device4_machine1.dp_gflops, 7800)
        self.assertAlmostEqual(device4_machine1.memory, 32)
        link0_machine1 = machine1.links[5, 6]
        self.assertEqual(link0_machine1.source.global_id, 5)
        self.assertEqual(link0_machine1.target.global_id, 6)
        self.assertEqual(link0_machine1.type, LinkType.PHB)
        self.assertAlmostEqual(link0_machine1.bandwidth, 12)
        self.assertAlmostEqual(link0_machine1.latency, 0)
        link1_machine1 = machine1.links[5, 7]
        self.assertEqual(link1_machine1.source.global_id, 5)
        self.assertEqual(link1_machine1.target.global_id, 7)
        self.assertEqual(link1_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link1_machine1.bandwidth, 1)
        self.assertAlmostEqual(link1_machine1.latency, 0)
        device6_machine1 = machine1.devices[6]
        self.assertEqual(device6_machine1.global_id, 6)
        self.assertEqual(device6_machine1.local_id, 0)
        self.assertEqual(device6_machine1.type, DeviceType.CPU)
        self.assertEqual(device6_machine1.model, 'Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G')
        self.assertAlmostEqual(device6_machine1.sp_gflops, 150)
        self.assertAlmostEqual(device6_machine1.dp_gflops, 75)
        self.assertAlmostEqual(device6_machine1.memory, 503)
        link0_machine1 = machine1.links[6, 7]
        self.assertEqual(link0_machine1.source.global_id, 6)
        self.assertEqual(link0_machine1.target.global_id, 7)
        self.assertEqual(link0_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine1.bandwidth, 1)
        self.assertAlmostEqual(link0_machine1.latency, 0)
        device7_machine1 = machine1.devices[7]
        self.assertEqual(device7_machine1.global_id, 7)
        self.assertEqual(device7_machine1.local_id, 0)
        self.assertEqual(device7_machine1.type, DeviceType.NIC)
        self.assertAlmostEqual(device7_machine1.model, None)
        self.assertAlmostEqual(device7_machine1.sp_gflops, 0)
        self.assertAlmostEqual(device7_machine1.dp_gflops, 0)
        self.assertAlmostEqual(device7_machine1.memory, 0)
        link0_machine1 = machine1.links[7, 3]
        self.assertEqual(link0_machine1.source.global_id, 7)
        self.assertEqual(link0_machine1.target.global_id, 3)
        self.assertEqual(link0_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine1.bandwidth, 1)
        self.assertAlmostEqual(link0_machine1.latency, 0)
        str = f'cluster: {cluster}'
if __name__ == '__main__':
    unittest.main()