import os
import torch
import kornia
from kornia.testing import assert_close

class TestSaveLoadPointCloud:

    def test_save_pointcloud(self):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = (10, 8)
        xyz_save = torch.rand(height, width, 3)
        filename = 'pointcloud.ply'
        kornia.utils.save_pointcloud_ply(filename, xyz_save)
        xyz_load = kornia.utils.load_pointcloud_ply(filename)
        assert_close(xyz_save.reshape(-1, 3), xyz_load)
        if os.path.exists(filename):
            os.remove(filename)

    @staticmethod
    def test_inf_coordinates_save_pointcloud():
        if False:
            i = 10
            return i + 15
        (height, width) = (10, 8)
        xyz_save = torch.rand(height, width, 3)
        xyz_save[0, 0, :] = float('inf')
        xyz_save[0, 1, 0] = float('inf')
        xyz_save[1, 0, :-1] = float('inf')
        filename = 'pointcloud.ply'
        kornia.utils.save_pointcloud_ply(filename, xyz_save)
        xyz_correct = xyz_save.reshape(-1, 3)[1:, :]
        xyz_load = kornia.utils.load_pointcloud_ply(filename)
        assert_close(xyz_correct, xyz_load)
        if os.path.exists(filename):
            os.remove(filename)