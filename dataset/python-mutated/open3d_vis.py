import copy
import numpy as np
import torch
try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError('Please run "pip install open3d" to install open3d first.')

def _draw_points(points, vis, points_size=2, point_color=(0.5, 0.5, 0.5), mode='xyz'):
    if False:
        print('Hello World!')
    "Draw points on visualizer.\n\n    Args:\n        points (numpy.array | torch.tensor, shape=[N, 3+C]):\n            points to visualize.\n        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.\n        points_size (int, optional): the size of points to show on visualizer.\n            Default: 2.\n        point_color (tuple[float], optional): the color of points.\n            Default: (0.5, 0.5, 0.5).\n        mode (str, optional):  indicate type of the input points,\n            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.\n\n    Returns:\n        tuple: points, color of each point.\n    "
    vis.get_render_option().point_size = points_size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)
    return (pcd, points_colors)

def _draw_bboxes(bbox3d, vis, points_colors, pcd=None, bbox_color=(0, 1, 0), points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
    if False:
        for i in range(10):
            print('nop')
    "Draw bbox on visualizer and change the color of points inside bbox3d.\n\n    Args:\n        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):\n            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.\n        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.\n        points_colors (numpy.array): color of each points.\n        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.\n            Default: None.\n        bbox_color (tuple[float], optional): the color of bbox.\n            Default: (0, 1, 0).\n        points_in_box_color (tuple[float], optional):\n            the color of points inside bbox3d. Default: (1, 0, 0).\n        rot_axis (int, optional): rotation axis of bbox. Default: 2.\n        center_mode (bool, optional): indicate the center of bbox is\n            bottom center or gravity center. available mode\n            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.\n        mode (str, optional):  indicate type of the input points,\n            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.\n    "
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()
    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[rot_axis] / 2
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[rot_axis] / 2
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        vis.add_geometry(line_set)
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)

def show_pts_boxes(points, bbox3d=None, show=True, save_path=None, points_size=2, point_color=(0.5, 0.5, 0.5), bbox_color=(0, 1, 0), points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
    if False:
        return 10
    "Draw bbox and points on visualizer.\n\n    Args:\n        points (numpy.array | torch.tensor, shape=[N, 3+C]):\n            points to visualize.\n        bbox3d (numpy.array | torch.tensor, shape=[M, 7], optional):\n            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.\n            Defaults to None.\n        show (bool, optional): whether to show the visualization results.\n            Default: True.\n        save_path (str, optional): path to save visualized results.\n            Default: None.\n        points_size (int, optional): the size of points to show on visualizer.\n            Default: 2.\n        point_color (tuple[float], optional): the color of points.\n            Default: (0.5, 0.5, 0.5).\n        bbox_color (tuple[float], optional): the color of bbox.\n            Default: (0, 1, 0).\n        points_in_box_color (tuple[float], optional):\n            the color of points which are in bbox3d. Default: (1, 0, 0).\n        rot_axis (int, optional): rotation axis of bbox. Default: 2.\n        center_mode (bool, optional): indicate the center of bbox is bottom\n            center or gravity center. available mode\n            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.\n        mode (str, optional):  indicate type of the input points, available\n            mode ['xyz', 'xyzrgb']. Default: 'xyz'.\n    "
    assert 0 <= rot_axis <= 2
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    (pcd, points_colors) = _draw_points(points, vis, points_size, point_color, mode)
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color, points_in_box_color, rot_axis, center_mode, mode)
    if show:
        vis.run()
    if save_path is not None:
        vis.capture_screen_image(save_path)
    vis.destroy_window()

def _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd=None, bbox_color=(0, 1, 0), points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
    if False:
        while True:
            i = 10
    "Draw bbox on visualizer and change the color or points inside bbox3d\n    with indices.\n\n    Args:\n        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):\n            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.\n        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.\n        indices (numpy.array | torch.tensor, shape=[N, M]):\n            indicate which bbox3d that each point lies in.\n        points_colors (numpy.array): color of each points.\n        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.\n            Default: None.\n        bbox_color (tuple[float], optional): the color of bbox.\n            Default: (0, 1, 0).\n        points_in_box_color (tuple[float], optional):\n            the color of points which are in bbox3d. Default: (1, 0, 0).\n        rot_axis (int, optional): rotation axis of bbox. Default: 2.\n        center_mode (bool, optional): indicate the center of bbox is\n            bottom center or gravity center. available mode\n            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.\n        mode (str, optional):  indicate type of the input points,\n            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.\n    "
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    bbox3d = bbox3d.copy()
    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[rot_axis] / 2
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[rot_axis] / 2
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        vis.add_geometry(line_set)
        if pcd is not None and mode == 'xyz':
            points_colors[indices[:, i].astype(np.bool)] = in_box_color
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)

def show_pts_index_boxes(points, bbox3d=None, show=True, indices=None, save_path=None, points_size=2, point_color=(0.5, 0.5, 0.5), bbox_color=(0, 1, 0), points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
    if False:
        print('Hello World!')
    "Draw bbox and points on visualizer with indices that indicate which\n    bbox3d that each point lies in.\n\n    Args:\n        points (numpy.array | torch.tensor, shape=[N, 3+C]):\n            points to visualize.\n        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):\n            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.\n            Defaults to None.\n        show (bool, optional): whether to show the visualization results.\n            Default: True.\n        indices (numpy.array | torch.tensor, shape=[N, M], optional):\n            indicate which bbox3d that each point lies in. Default: None.\n        save_path (str, optional): path to save visualized results.\n            Default: None.\n        points_size (int, optional): the size of points to show on visualizer.\n            Default: 2.\n        point_color (tuple[float], optional): the color of points.\n            Default: (0.5, 0.5, 0.5).\n        bbox_color (tuple[float], optional): the color of bbox.\n            Default: (0, 1, 0).\n        points_in_box_color (tuple[float], optional):\n            the color of points which are in bbox3d. Default: (1, 0, 0).\n        rot_axis (int, optional): rotation axis of bbox. Default: 2.\n        center_mode (bool, optional): indicate the center of bbox is\n            bottom center or gravity center. available mode\n            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.\n        mode (str, optional):  indicate type of the input points,\n            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.\n    "
    assert 0 <= rot_axis <= 2
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    (pcd, points_colors) = _draw_points(points, vis, points_size, point_color, mode)
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color, points_in_box_color, rot_axis, center_mode, mode)
    if show:
        vis.run()
    if save_path is not None:
        vis.capture_screen_image(save_path)
    vis.destroy_window()

class Visualizer(object):
    """Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self, points, bbox3d=None, save_path=None, points_size=2, point_color=(0.5, 0.5, 0.5), bbox_color=(0, 1, 0), points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
        if False:
            i = 10
            return i + 15
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.o3d_visualizer.add_geometry(mesh_frame)
        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0
        if points is not None:
            (self.pcd, self.points_colors) = _draw_points(points, self.o3d_visualizer, points_size, point_color, mode)
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd, bbox_color, points_in_box_color, rot_axis, center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        if False:
            return 10
        'Add bounding box to visualizer.\n\n        Args:\n            bbox3d (numpy.array, shape=[M, 7]):\n                3D bbox (x, y, z, x_size, y_size, z_size, yaw)\n                to be visualized. The 3d bbox is in mode of\n                Box3DMode.DEPTH with gravity_center (please refer to\n                core.structures.box_3d_mode).\n            bbox_color (tuple[float]): the color of bbox. Default: None.\n            points_in_box_color (tuple[float]): the color of points which\n                are in bbox3d. Default: None.\n        '
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd, bbox_color, points_in_box_color, self.rot_axis, self.center_mode, self.mode)

    def add_seg_mask(self, seg_mask_colors):
        if False:
            return 10
        'Add segmentation mask to visualizer via per-point colorization.\n\n        Args:\n            seg_mask_colors (numpy.array, shape=[N, 6]):\n                The segmentation mask whose first 3 dims are point coordinates\n                and last 3 dims are converted colors.\n        '
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) - np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[offset, 0, 0])
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        if False:
            print('Hello World!')
        'Visualize the points cloud.\n\n        Args:\n            save_path (str, optional): path to save image. Default: None.\n        '
        self.o3d_visualizer.run()
        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)
        self.o3d_visualizer.destroy_window()
        return