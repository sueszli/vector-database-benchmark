import os
import torch

def save_pointcloud_ply(filename: str, pointcloud: torch.Tensor) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Utility function to save to disk a pointcloud in PLY format.\n\n    Args:\n        filename: the path to save the pointcloud.\n        pointcloud: tensor containing the pointcloud to save.\n          The tensor must be in the shape of :math:`(*, 3)` where the last\n          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.\n    '
    if not isinstance(filename, str) and filename[-3:] == '.ply':
        raise TypeError(f'Input filename must be a string in with the .ply  extension. Got {filename}')
    if not torch.is_tensor(pointcloud):
        raise TypeError(f'Input pointcloud type is not a torch.Tensor. Got {type(pointcloud)}')
    if not len(pointcloud.shape) >= 2 and pointcloud.shape[-1] == 3:
        raise TypeError(f'Input pointcloud must be in the following shape HxWx3. Got {pointcloud.shape}.')
    xyz_vec: torch.Tensor = pointcloud.reshape(-1, 3)
    with open(filename, 'w') as f:
        data_str: str = ''
        num_points: int = xyz_vec.shape[0]
        for idx in range(num_points):
            xyz = xyz_vec[idx]
            if not bool(torch.isfinite(xyz).any()):
                num_points -= 1
                continue
            x: float = float(xyz[0])
            y: float = float(xyz[1])
            z: float = float(xyz[2])
            data_str += f'{x} {y} {z}\n'
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment arraiy generated\n')
        f.write('element vertex %d\n' % num_points)
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('end_header\n')
        f.write(data_str)

def load_pointcloud_ply(filename: str, header_size: int=8) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Utility function to load from disk a pointcloud in PLY format.\n\n    Args:\n        filename: the path to the pointcloud.\n        header_size: the size of the ply file header that will\n          be skipped during loading.\n\n    Return:\n        tensor containing the loaded point with shape :math:`(*, 3)` where\n        :math:`*` represents the number of points.\n    '
    if not isinstance(filename, str) and filename[-3:] == '.ply':
        raise TypeError(f'Input filename must be a string in with the .ply  extension. Got {filename}')
    if not os.path.isfile(filename):
        raise ValueError('Input filename is not an existing file.')
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError(f'Input header_size must be a positive integer. Got {header_size}.')
    with open(filename) as f:
        points = []
        lines = f.readlines()[header_size:]
        for line in lines:
            (x_str, y_str, z_str) = line.split()
            points.append((torch.tensor(float(x_str)), torch.tensor(float(y_str)), torch.tensor(float(z_str))))
        pointcloud: torch.Tensor = torch.tensor(points)
        return pointcloud