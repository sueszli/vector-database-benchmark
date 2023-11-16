import torch
from modelscope.models.cv.video_depth_estimation.geometry.pose_utils import invert_pose, pose_vec2mat

class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """

    def __init__(self, mat):
        if False:
            return 10
        '\n        Initializes a Pose object.\n\n        Parameters\n        ----------\n        mat : torch.Tensor [B,4,4]\n            Transformation matrix\n        '
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        if False:
            while True:
                i = 10
        'Batch size of the transformation matrix'
        return len(self.mat)

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        if False:
            return 10
        'Initializes as a [4,4] identity matrix'
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N, 1, 1]))

    @classmethod
    def from_vec(cls, vec, mode):
        if False:
            return 10
        'Initializes from a [B,6] batch vector'
        mat = pose_vec2mat(vec, mode)
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        'Returns the transformation matrix shape'
        return self.mat.shape

    def item(self):
        if False:
            while True:
                i = 10
        'Returns the transformation matrix'
        return self.mat

    def repeat(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Repeats the transformation matrix multiple times'
        self.mat = self.mat.repeat(*args, **kwargs)
        return self

    def inverse(self):
        if False:
            while True:
                i = 10
        'Returns a new Pose that is the inverse of this one'
        return Pose(invert_pose(self.mat))

    def to(self, *args, **kwargs):
        if False:
            return 10
        'Moves object to a specific device'
        self.mat = self.mat.to(*args, **kwargs)
        return self

    def transform_pose(self, pose):
        if False:
            i = 10
            return i + 15
        'Creates a new pose object that compounds this and another one (self * pose)'
        assert tuple(pose.shape[-2:]) == (4, 4)
        return Pose(self.mat.bmm(pose.item()))

    def transform_points(self, points):
        if False:
            print('Hello World!')
        'Transforms 3D points using this object'
        assert points.shape[1] == 3
        (B, _, H, W) = points.shape
        out = self.mat[:, :3, :3].bmm(points.view(B, 3, -1)) + self.mat[:, :3, -1].unsqueeze(-1)
        return out.view(B, 3, H, W)

    def __matmul__(self, other):
        if False:
            while True:
                i = 10
        'Transforms the input (Pose or 3D points) using this object'
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()