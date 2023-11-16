import unittest

class TestConvertToProcessMeshes(unittest.TestCase):

    def test_convert_to_process_meshes(self):
        if False:
            return 10
        device_meshes = [[1, 8], [4, 8], [15, 8]]
        from paddle.distributed.auto_parallel.static.tuner.rule_based_tuner import convert_to_process_meshes
        process_meshes = []
        for device_mesh in device_meshes:
            process_mesh = convert_to_process_meshes(device_mesh)
            process_meshes.append(process_mesh)
if __name__ == '__main__':
    unittest.main()