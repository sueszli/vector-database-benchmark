import copy
from opensfm.pymap import RigCamera, RigInstance, Shot
from opensfm.types import Reconstruction
from typing import Tuple
import random
import numpy as np
import pytest
from opensfm import pygeometry
from opensfm import pymap
from opensfm import types
from opensfm.test.utils import assert_maps_equal, assert_metadata_equal, assert_cameras_equal, assert_shots_equal

def _create_reconstruction(n_cameras: int=0, n_shots_cam=None, n_pano_shots_cam=None, n_points: int=0, dist_to_shots: bool=False, dist_to_pano_shots: bool=False) -> types.Reconstruction:
    if False:
        print('Hello World!')
    'Creates a reconstruction with n_cameras random cameras and\n    shots, where n_shots_cam is a dictionary, containing the\n    camera_id and the number of shots.\n\n    Example:\n    shot_cams = {"0": 50, "1": 30}\n    _create_reconstruction(2, shot_cams)\n\n    Will create a reconstruction with two cameras and 80 shots,\n    50 are associated with cam "0" and 30 with cam "1".\n\n    n_points_in_shots is the number of points to create.\n    If dist_to_shots, then observations are created and randomly\n    distributed to all shots. We pick with the repeat option, thus\n    if we have three shots the distribution could be\n    something like: [1,2,2], [0,1,2]. We avoid things like [3,3,3]\n    '
    if n_shots_cam is None:
        n_shots_cam = {}
    if n_pano_shots_cam is None:
        n_pano_shots_cam = {}
    rec = types.Reconstruction()
    if n_cameras > 0:
        for i in range(n_cameras):
            (focal, k1, k2) = np.random.rand(3)
            cam = pygeometry.Camera.create_perspective(focal, k1, k2)
            cam.id = str(i)
            rec.add_camera(cam)
        shot_id = 0
        for (cam_id, n_shots) in n_shots_cam.items():
            for _ in range(n_shots):
                rec.create_shot(str(shot_id), cam_id)
                shot_id += 1
        shot_id = 0
        for (cam_id, n_shots) in n_pano_shots_cam.items():
            for _ in range(n_shots):
                rec.create_pano_shot(str(shot_id), cam_id)
                shot_id += 1
    if n_points > 0:
        for i in range(n_points):
            rec.create_point(str(i), np.random.rand(3))
        if dist_to_shots:
            n_shots = len(rec.shots)
            for pt in rec.points.values():
                choice = set(np.random.choice(n_shots, n_shots))
                if len(choice) > 1:
                    for ch in choice:
                        obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, int(pt.id))
                        shot = rec.shots[str(ch)]
                        rec.add_observation(shot, pt, obs)
    return rec
'\nCamera Tests\n'

def test_create_cameras() -> None:
    if False:
        i = 10
        return i + 15
    n_cameras = 100
    rec = types.Reconstruction()
    for cam_id in range(0, n_cameras):
        (focal, k1, k2) = np.random.rand(3)
        cam = pygeometry.Camera.create_perspective(focal, k1, k2)
        cam.id = str(cam_id)
        map_cam = rec.add_camera(cam)
        assert_cameras_equal(cam, map_cam)
        assert cam is not map_cam
        assert map_cam is rec.get_camera(str(cam_id))
        assert map_cam is rec.cameras[str(cam_id)]
    assert len(rec.cameras) == n_cameras

def test_camera_iterators() -> None:
    if False:
        while True:
            i = 10
    n_cameras = 100
    rec = _create_reconstruction(n_cameras)
    visited_cams = set()
    for cam_id in rec.cameras:
        visited_cams.add(cam_id)
    assert len(visited_cams) == n_cameras
    for idx in range(0, n_cameras):
        assert str(idx) in visited_cams
    visited_cams = set()
    for cam in rec.cameras.values():
        visited_cams.add(cam.id)
        focal = np.random.rand(1)
        cam.focal = focal
        assert rec.cameras[cam.id].focal == focal
        assert cam is rec.cameras[cam.id]
    assert len(visited_cams) == n_cameras
    for idx in range(0, n_cameras):
        assert str(idx) in visited_cams
    for (cam_id, cam) in rec.cameras.items():
        assert cam_id == cam.id
        focal = np.random.rand(1)
        cam.focal = focal
        assert rec.cameras[cam.id].focal == focal
        assert cam is rec.cameras[cam.id]

def _check_common_cam_properties(cam1, cam2) -> None:
    if False:
        print('Hello World!')
    assert cam1.id == cam2.id
    assert cam1.width == cam2.width
    assert cam1.height == cam2.height
    assert cam1.projection_type == cam2.projection_type

def test_brown_camera() -> None:
    if False:
        i = 10
        return i + 15
    rec = types.Reconstruction()
    focal_x = 0.6
    focal_y = 0.7
    c_x = 0.1
    c_y = -0.05
    k1 = -0.1
    k2 = 0.01
    p1 = 0.001
    p2 = 0.002
    k3 = 0.01
    cam_cpp = pygeometry.Camera.create_brown(focal_x, focal_y / focal_x, np.array([c_x, c_y]), np.array([k1, k2, k3, p1, p2]))
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2 and (cam_cpp.k3 == c.k3)
    assert cam_cpp.p2 == c.p2 and cam_cpp.p1 == c.p1
    assert np.allclose(cam_cpp.principal_point, c.principal_point)
    assert len(c.distortion) == 5
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio

def test_fisheye_camera() -> None:
    if False:
        print('Hello World!')
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    cam_cpp = pygeometry.Camera.create_fisheye(focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal

def test_fisheye_opencv_camera() -> None:
    if False:
        while True:
            i = 10
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array([-0.1, 0.09, 0.08, 0.01])
    cam_cpp = pygeometry.Camera.create_fisheye_opencv(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio

def test_fisheye62_camera() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array([-0.1, 0.09, 0.08, 0.01, 0.02, 0.05, 0.1, 0.2])
    cam_cpp = pygeometry.Camera.create_fisheye62(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert cam_cpp.k5 == c.k5 and cam_cpp.k6 == c.k6
    assert cam_cpp.p1 == c.p1 and cam_cpp.p2 == c.p2
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio

def test_fisheye624_camera() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    focal = 0.6
    aspect_ratio = 0.7
    ppoint = np.array([0.51, 0.52])
    dist = np.array([-0.1, 0.09, 0.08, 0.01, 0.02, 0.05, 0.1, 0.2, 0.01, -0.003, 0.005, -0.007])
    cam_cpp = pygeometry.Camera.create_fisheye624(focal, aspect_ratio, ppoint, dist)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert cam_cpp.k3 == c.k3 and cam_cpp.k4 == c.k4
    assert cam_cpp.k5 == c.k5 and cam_cpp.k6 == c.k6
    assert cam_cpp.p1 == c.p1 and cam_cpp.p2 == c.p2
    assert cam_cpp.s0 == c.s0 and cam_cpp.s1 == c.s1
    assert cam_cpp.s2 == c.s2 and cam_cpp.s3 == c.s3
    assert len(dist) == len(c.distortion)
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.aspect_ratio == c.aspect_ratio

def test_dual_camera() -> None:
    if False:
        return 10
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    transition = 0.5
    cam_cpp = pygeometry.Camera.create_dual(transition, focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal
    assert cam_cpp.transition == c.transition

def test_perspective_camera() -> None:
    if False:
        i = 10
        return i + 15
    rec = types.Reconstruction()
    focal = 0.6
    k1 = -0.1
    k2 = 0.01
    cam_cpp = pygeometry.Camera.create_perspective(focal, k1, k2)
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)
    assert cam_cpp.k1 == c.k1 and cam_cpp.k2 == c.k2
    assert len(c.distortion) == 2
    assert np.allclose(cam_cpp.distortion, c.distortion)
    assert cam_cpp.focal == c.focal

def test_spherical_camera() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    cam_cpp = pygeometry.Camera.create_spherical()
    cam_cpp.width = 800
    cam_cpp.height = 600
    cam_cpp.id = 'cam'
    c = rec.add_camera(cam_cpp)
    _check_common_cam_properties(cam_cpp, c)

def _help_measurement_test(measurement, attr, val) -> None:
    if False:
        return 10
    assert getattr(measurement, attr).has_value is False
    getattr(measurement, attr).value = val
    if np.shape(val) == ():
        assert getattr(measurement, attr).value == val
    else:
        assert np.allclose(getattr(measurement, attr).value, val)
    assert getattr(measurement, attr).has_value is True
    getattr(measurement, attr).reset()
    assert getattr(measurement, attr).has_value is False

def test_shot_measurement_setter_and_getter() -> None:
    if False:
        i = 10
        return i + 15
    m1 = pymap.ShotMeasurements()
    _help_measurement_test(m1, 'capture_time', np.random.rand(1))
    _help_measurement_test(m1, 'gps_position', np.random.rand(3))
    _help_measurement_test(m1, 'gps_accuracy', np.random.rand(1))
    _help_measurement_test(m1, 'compass_accuracy', np.random.rand(1))
    _help_measurement_test(m1, 'compass_angle', np.random.rand(1))
    _help_measurement_test(m1, 'opk_accuracy', np.random.rand(1))
    _help_measurement_test(m1, 'opk_angles', np.random.rand(3))
    _help_measurement_test(m1, 'gravity_down', np.random.rand(3))
    _help_measurement_test(m1, 'orientation', random.randint(0, 100))
    _help_measurement_test(m1, 'sequence_key', 'key_test')

def _helper_populate_metadata(m) -> None:
    if False:
        print('Hello World!')
    m.capture_time.value = np.random.rand(1)
    m.gps_position.value = np.random.rand(3)
    m.gps_accuracy.value = np.random.rand(1)
    m.compass_accuracy.value = np.random.rand(1)
    m.compass_angle.value = np.random.rand(1)
    m.opk_accuracy.value = np.random.rand(1)
    m.opk_angles.value = np.random.rand(3)
    m.gravity_down.value = np.random.rand(3)
    m.orientation.value = random.randint(0, 100)
    m.sequence_key.value = 'sequence_key'

def test_shot_measurement_set() -> None:
    if False:
        i = 10
        return i + 15
    m1 = pymap.ShotMeasurements()
    _helper_populate_metadata(m1)
    m2 = pymap.ShotMeasurements()
    m2.set(m1)
    assert_metadata_equal(m1, m2)
    m3 = pymap.ShotMeasurements()
    m1.set(m3)
    assert_metadata_equal(m1, m3)

def test_shot_create() -> None:
    if False:
        print('Hello World!')
    rec = _create_reconstruction(2)
    shot1 = rec.create_shot('shot0', '0')
    assert shot1.id == 'shot0'
    assert shot1.camera.id == '0'
    assert len(rec.shots) == 1

def test_shot_create_existing() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(2)
    rec.create_shot('shot0', '0')
    with pytest.raises(RuntimeError):
        rec.create_shot('shot0', '0')
        rec.create_shot('shot0', '1')

def test_shot_create_more() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(2)
    rec.create_shot('shot0', '0')
    n_shots = 10
    for i in range(1, n_shots):
        rec.create_shot('shot' + str(i), '0')
    assert len(rec.shots) == n_shots

def test_shot_delete_non_existing() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(2)
    rec.create_shot('shot0', '0')
    with pytest.raises(RuntimeError):
        rec.remove_shot('abcde')

def test_shot_delete_existing() -> None:
    if False:
        for i in range(10):
            print('nop')
    n_shots = 10
    rec = _create_reconstruction(1, {'0': n_shots})
    del_shots = np.random.choice(n_shots, int(n_shots / 2), replace=False)
    for i in del_shots:
        rec.remove_shot(str(i))
    assert len(rec.shots) == n_shots - len(del_shots)

def test_shot_get() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = _create_reconstruction(1)
    shot_id = 'shot0'
    shot1 = rec.create_shot(shot_id, '0')
    assert shot1 is rec.get_shot(shot_id)
    assert shot1 is rec.shots[shot_id]

def test_shot_pose_set() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1)
    shot_id = 'shot0'
    shot = rec.create_shot(shot_id, '0')
    origin = np.array([1, 2, 3])
    shot.pose.set_origin(origin)
    assert np.allclose(origin, shot.pose.get_origin())

def test_shot_get_non_existing() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1)
    shot_id = 'shot0'
    shot1 = rec.create_shot(shot_id, '0')
    with pytest.raises(RuntimeError):
        assert shot1 is rec.get_shot('toto')
    with pytest.raises(RuntimeError):
        assert shot1 is rec.shots['toto']

def test_pano_shot_get() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(1)
    shot_id = 'shot0'
    shot1 = rec.create_pano_shot(shot_id, '0')
    assert shot1 is rec.pano_shots[shot_id]
    assert shot1 is rec.get_pano_shot(shot_id)

def test_pano_shot_get_non_existing() -> None:
    if False:
        return 10
    rec = _create_reconstruction(1)
    shot_id = 'shot0'
    shot1 = rec.create_shot(shot_id, '0')
    with pytest.raises(RuntimeError):
        assert shot1 is rec.get_shot('toto')
    with pytest.raises(RuntimeError):
        assert shot1 is rec.shots['toto']

def test_pano_shot_create() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = _create_reconstruction(2)
    shot1 = rec.create_pano_shot('shot0', '0')
    assert shot1.id == 'shot0'
    assert shot1.camera.id == '0'
    assert len(rec.pano_shots) == 1

def test_pano_shot_create_existing() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(2)
    rec.create_pano_shot('shot0', '0')
    n_shots = 10
    for _ in range(n_shots):
        with pytest.raises(RuntimeError):
            rec.create_pano_shot('shot0', '0')
            rec.create_pano_shot('shot0', '1')

def test_pano_shot_create_more() -> None:
    if False:
        return 10
    rec = _create_reconstruction(2)
    rec.create_pano_shot('shot0', '0')
    n_shots = 10
    for i in range(1, n_shots):
        rec.create_pano_shot('shot' + str(i), '0')
    assert len(rec.pano_shots) == n_shots

def test_pano_shot_delete_non_existing() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(2)
    rec.create_pano_shot('shot0', '0')
    with pytest.raises(RuntimeError):
        rec.remove_pano_shot('abcde')

def test_pano_shot_delete_existing() -> None:
    if False:
        print('Hello World!')
    n_shots = 10
    rec = _create_reconstruction(2)
    rec = _create_reconstruction(1, n_pano_shots_cam={'0': n_shots})
    n_shots = 10
    del_shots = np.random.choice(n_shots, int(n_shots / 2), replace=False)
    for i in del_shots:
        rec.remove_pano_shot(str(i))
    assert len(rec.pano_shots) == n_shots - len(del_shots)

def test_shot_merge_cc() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1, {'0': 2})
    map_shot1 = rec.shots['0']
    map_shot1.merge_cc = 10
    assert map_shot1.merge_cc == 10

def test_shot_covariance() -> None:
    if False:
        print('Hello World!')
    rec = _create_reconstruction(1, {'0': 2})
    map_shot1 = rec.shots['0']
    map_shot1.covariance = np.diag([1, 2, 3])
    assert np.allclose(map_shot1.covariance, np.diag([1, 2, 3]))

def test_shot_covariance_different() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1, {'0': 2})
    map_shot1 = rec.shots['0']
    map_shot2 = rec.shots['1']
    map_shot1.covariance = np.diag([1, 2, 3])
    map_shot2.covariance = np.diag([2, 2, 2])
    assert map_shot2.covariance is not map_shot1.covariance

def test_shot_create_remove_create() -> None:
    if False:
        return 10
    n_shots = 10
    rec = _create_reconstruction(1, {'0': n_shots})
    rec.remove_shot('0')
    assert len(rec.shots) == n_shots - 1
    rec.create_shot('0', '0')
    assert len(rec.shots) == n_shots

def test_pano_shot_create_remove_create() -> None:
    if False:
        return 10
    n_shots = 10
    rec = _create_reconstruction(1, n_pano_shots_cam={'0': n_shots})
    rec.remove_pano_shot('0')
    assert len(rec.pano_shots) == n_shots - 1
    rec.create_pano_shot('0', '0')
    assert len(rec.pano_shots) == n_shots

def _create_rig_camera() -> RigCamera:
    if False:
        print('Hello World!')
    rig_camera = pymap.RigCamera()
    rig_camera.id = 'rig_camera'
    rig_camera.pose = pygeometry.Pose(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
    return rig_camera

def _create_rig_instance() -> Tuple[Reconstruction, RigInstance, Shot]:
    if False:
        return 10
    rec = _create_reconstruction(1, {'0': 2})
    rig_camera = rec.add_rig_camera(_create_rig_camera())
    rig_instance = pymap.RigInstance('1')
    shot = pymap.Shot('0', pygeometry.Camera.create_spherical(), pygeometry.Pose())
    rig_instance.add_shot(rig_camera, shot)
    return (rec, rig_instance, shot)

def test_rig_camera_create() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(1, {'0': 2})
    rec.add_rig_camera(_create_rig_camera())
    assert '0' in rec.rig_cameras.keys()
    assert 'rig_camera' in rec.rig_cameras.keys()

def test_rig_instance() -> None:
    if False:
        print('Hello World!')
    (_, rig_instance, _) = _create_rig_instance()
    assert list(rig_instance.keys()) == ['0']

def test_rig_instance_create_default() -> None:
    if False:
        i = 10
        return i + 15
    (rec, rig_instance, _) = _create_rig_instance()
    assert len(rec.rig_instances) == 2
    assert dict(rec.rig_instances['0'].camera_ids.items()) == {'0': '0'}
    assert list(rec.rig_instances['0'].shots.keys()) == ['0']
    assert dict(rec.rig_instances['1'].camera_ids.items()) == {'1': '0'}
    assert list(rec.rig_instances['1'].shots.keys()) == ['1']

def test_rig_instance_create_add_existing() -> None:
    if False:
        return 10
    (rec, rig_instance, _) = _create_rig_instance()
    with pytest.raises(RuntimeError):
        rec.add_rig_instance(rig_instance)

def test_rig_instance_remove_shot() -> None:
    if False:
        return 10
    (rec, _, shot) = _create_rig_instance()
    rec.remove_shot(shot.id)
    assert len(rec.rig_instances['0'].shots) == 0

def test_rig_shot_modify_pose_raise() -> None:
    if False:
        return 10
    (_, rig_instance, shot) = _create_rig_instance()
    with pytest.raises(RuntimeError):
        shot.pose.set_origin(np.array([1, 2, 3]))

def test_rig_shot_modify_pose_succeed() -> None:
    if False:
        i = 10
        return i + 15
    (_, rig_instance, shot) = _create_rig_instance()
    next(iter(rig_instance.rig_cameras.values())).pose = pygeometry.Pose()
    shot.pose.set_origin(np.array([1, 2, 3]))

def test_rig_shot_set_pose() -> None:
    if False:
        for i in range(10):
            print('nop')
    (_, rig_instance, shot) = _create_rig_instance()
    with pytest.raises(RuntimeError):
        shot.pose = pygeometry.Pose()

def test_add_shot_from_shot_correct_value() -> None:
    if False:
        i = 10
        return i + 15
    n_shots = 5
    rec = _create_reconstruction(1, n_shots_cam={'0': n_shots})
    shot1 = rec.shots['0']
    _helper_populate_metadata(shot1.metadata)
    rec_new = _create_reconstruction(1)
    rec_new.add_shot(rec.shots['0'])
    rec_new.add_shot(rec.shots['1'])
    assert len(rec_new.shots) == 2
    for k in rec_new.shots.keys():
        assert_shots_equal(rec.shots[k], rec_new.shots[k])

def test_shot_metadata_different() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1, n_shots_cam={'0': 2})
    shot1 = rec.shots['0']
    shot2 = rec.shots['1']
    _helper_populate_metadata(shot1.metadata)
    assert shot1.metadata is not shot2.metadata

def test_shot_metadata_assign_equal() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1, n_shots_cam={'0': 2})
    shot1 = rec.shots['0']
    shot2 = rec.shots['1']
    _helper_populate_metadata(shot1.metadata)
    shot2.metadata = shot1.metadata
    assert shot1.metadata is not shot2.metadata
    assert_metadata_equal(shot1.metadata, shot2.metadata)

def test_add_pano_shot_from_shot_correct_value() -> None:
    if False:
        print('Hello World!')
    n_shots = 5
    rec = _create_reconstruction(1, n_pano_shots_cam={'0': n_shots})
    shot1 = rec.pano_shots['0']
    _helper_populate_metadata(shot1.metadata)
    rec_new = _create_reconstruction(1)
    rec_new.add_pano_shot(rec.pano_shots['0'])
    rec_new.add_pano_shot(rec.pano_shots['1'])
    for k in rec_new.shots.keys():
        assert_shots_equal(rec.pano_shots[k], rec_new.pano_shots[k])

def test_single_point_create() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    pt = rec.create_point('0')
    assert pt.id == '0'
    assert len(rec.points) == 1

def test_single_point_get_existing() -> None:
    if False:
        return 10
    rec = types.Reconstruction()
    pt = rec.create_point('0')
    assert pt == rec.points['0'] and pt == rec.get_point('0')

def test_single_point_get_non_existing() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    rec.create_point('0')
    with pytest.raises(RuntimeError):
        rec.get_point('toto')

def test_single_point_coordinates() -> None:
    if False:
        for i in range(10):
            print('nop')
    rec = types.Reconstruction()
    pt = rec.create_point('0')
    coord = np.random.rand(3)
    pt.coordinates = coord
    assert np.allclose(pt.coordinates, coord)

def test_single_point_color() -> None:
    if False:
        return 10
    rec = types.Reconstruction()
    pt = rec.create_point('0')
    color = np.random.randint(low=0, high=255, size=(3,))
    pt.color = color
    assert np.allclose(pt.color, color)

def test_point_add_from_point() -> None:
    if False:
        return 10
    rec = types.Reconstruction()
    rec2 = types.Reconstruction()
    coord2 = np.random.rand(3)
    pt2 = rec2.create_point('1', coord2)
    pt2_1 = rec.add_point(pt2)
    assert len(rec.points) == 1
    assert pt2 is not pt2_1
    assert '1' == pt2_1.id
    assert pt2_1 == rec.points['1']
    assert np.allclose(pt2_1.coordinates, coord2)

def test_point_reproj_errors_assign() -> None:
    if False:
        print('Hello World!')
    rec = _create_reconstruction(n_points=1)
    pt = rec.points['0']
    reproj_errors = dict({'shot1': np.random.rand(2), 'shot2': np.random.rand(2)})
    pt.reprojection_errors = reproj_errors
    for k in reproj_errors.keys():
        assert np.allclose(pt.reprojection_errors[k], reproj_errors[k])

def test_point_delete_non_existing() -> None:
    if False:
        print('Hello World!')
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)
    with pytest.raises(RuntimeError):
        rec.remove_point('abcdef')

def test_point_delete_existing() -> None:
    if False:
        print('Hello World!')
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)
    del_list = list(rec.points.keys())
    for k in del_list:
        rec.remove_point(k)
    assert len(rec.points) == 0

def test_point_delete_existing_assign_empty() -> None:
    if False:
        for i in range(10):
            print('nop')
    n_points = 100
    rec = _create_reconstruction(n_points=n_points)
    rec.points = {}
    assert len(rec.points) == 0

def test_single_observation() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(1, n_shots_cam={'0': 1}, n_points=1)
    obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100, 2, 5)
    rec.add_observation('0', '0', obs)
    shot = rec.shots['0']
    pt = rec.points['0']
    observations = pt.get_observations()
    assert len(observations) == 1
    assert pt.number_of_observations() == 1
    obs = shot.get_landmark_observation(pt)
    assert obs is not None

def test_single_observation_delete() -> None:
    if False:
        i = 10
        return i + 15
    rec = _create_reconstruction(1, n_shots_cam={'0': 1}, n_points=1)
    obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100)
    rec.add_observation('0', '0', obs)
    shot = rec.shots['0']
    pt = rec.points['0']
    rec.remove_observation(shot.id, pt.id)
    observations = pt.get_observations()
    assert len(observations) == 0
    assert pt.number_of_observations() == 0

def test_many_observations_delete() -> None:
    if False:
        i = 10
        return i + 15
    m = pymap.Map()
    n_cams = 2
    n_shots = 10
    n_landmarks = 1000
    for cam_id in range(n_cams):
        cam = pygeometry.Camera.create_perspective(0.5, 0, 0)
        cam.id = 'cam' + str(cam_id)
        m.create_camera(cam)
        m.create_rig_camera(pymap.RigCamera(pygeometry.Pose(), cam.id))
    for shot_id in range(n_shots):
        cam_id = 'cam' + str(int(np.random.rand(1) * 10 % n_cams))
        shot_id = str(shot_id)
        m.create_rig_instance(shot_id)
        m.create_shot(shot_id, cam_id, cam_id, shot_id, pygeometry.Pose())
    for point_id in range(n_landmarks):
        m.create_landmark(str(point_id), np.random.rand(3))
    n_total_obs = 0
    for lm in m.get_landmarks().values():
        n_obs = 0
        for shot in m.get_shots().values():
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, int(lm.id))
            m.add_observation(shot, lm, obs)
            n_obs += 1
            n_total_obs += 1
    for lm in m.get_landmarks().values():
        n_total_obs -= lm.number_of_observations()
    assert n_total_obs == 0
    m.clear_observations_and_landmarks()

def test_clean_landmarks_with_min_observations() -> None:
    if False:
        while True:
            i = 10
    m = pymap.Map()
    n_cams = 2
    n_shots = 2
    n_landmarks = 10
    for cam_id in range(n_cams):
        cam = pygeometry.Camera.create_perspective(0.5, 0, 0)
        cam.id = 'cam' + str(cam_id)
        m.create_camera(cam)
        m.create_rig_camera(pymap.RigCamera(pygeometry.Pose(), cam.id))
    for shot_id in range(n_shots):
        cam_id = 'cam' + str(int(np.random.rand(1) * 10 % n_cams))
        m.create_rig_instance(str(shot_id))
        m.create_shot(str(shot_id), cam_id, cam_id, str(shot_id), pygeometry.Pose())
    for point_id in range(n_landmarks):
        m.create_landmark(str(point_id), np.random.rand(3))
    for point_id in range(int(n_landmarks / 2)):
        for shot in m.get_shots().values():
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, point_id)
            m.add_observation(shot, m.get_landmark(str(point_id)), obs)
    for point_id in range(int(n_landmarks / 2), n_landmarks):
        shot = m.get_shot('0')
        obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, point_id)
        m.add_observation(shot, m.get_landmark(str(point_id)), obs)
    m.clean_landmarks_below_min_observations(n_shots)
    assert len(m.get_landmarks()) == int(n_landmarks / 2)
    m.clean_landmarks_below_min_observations(n_shots + 1)
    assert len(m.get_landmarks()) == 0

def test_camera_deepcopy() -> None:
    if False:
        i = 10
        return i + 15
    cam1 = pygeometry.Camera.create_perspective(0.5, 0, 0)
    cam2 = copy.deepcopy(cam1)
    assert cam1.focal == cam2.focal

def test_camera_deepcopy_assign() -> None:
    if False:
        i = 10
        return i + 15
    cam1 = pygeometry.Camera.create_perspective(0.5, 0, 0)
    cam2 = copy.deepcopy(cam1)
    cam2.focal = 0.7
    assert cam1.focal != cam2.focal

def test_observation_shot_removal() -> None:
    if False:
        return 10
    rec = _create_reconstruction(n_cameras=2, n_shots_cam={'0': 1, '1': 1}, n_points=200, dist_to_shots=True)
    rec.remove_shot('0')
    for p in rec.points.values():
        assert len(p.get_observations()) <= 1
    rec.remove_shot('1')
    for p in rec.points.values():
        assert len(p.get_observations()) == 0

def test_rec_deepcopy() -> None:
    if False:
        while True:
            i = 10
    rec = _create_reconstruction(n_cameras=2, n_shots_cam={'0': 50, '1': 40}, n_pano_shots_cam={'0': 20, '1': 30}, n_points=200, dist_to_shots=True)
    for shot in rec.shots.values():
        _helper_populate_metadata(shot.metadata)
    for shot in rec.pano_shots.values():
        _helper_populate_metadata(shot.metadata)
    rec2 = copy.deepcopy(rec, {'copy_observations': True})
    assert len(rec2.cameras) == 2
    assert len(rec2.shots) == 90
    assert len(rec2.pano_shots) == 50
    assert len(rec2.points) == 200
    assert_maps_equal(rec.map, rec2.map)

def test_gcp() -> None:
    if False:
        return 10
    gcp = []
    for i in range(0, 10):
        p = pymap.GroundControlPoint()
        p.id = 'p' + str(i)
        o1 = pymap.GroundControlPointObservation()
        o1.shot_id = 'p1'
        o2 = pymap.GroundControlPointObservation()
        o2.shot_id = 'p2'
        obs = [o1, o2]
        p.observations = obs
        gcp.append(p)
        assert p.observations[0].shot_id == 'p1'
        assert p.observations[1].shot_id == 'p2'
        p.add_observation(o2)
        p.add_observation(o2)
        assert len(p.observations) == 4
    for pt in gcp:
        assert pt.observations[0].shot_id == 'p1'
        assert pt.observations[1].shot_id == 'p2'

def test_add_correspondences_from_tracks_manager() -> None:
    if False:
        i = 10
        return i + 15
    n_shots = 3
    rec = _create_reconstruction(n_cameras=1, n_shots_cam={'0': n_shots}, n_points=10)
    tm = pymap.TracksManager()
    for track_id in ['0', '1', '100']:
        for shot_id in range(n_shots + 1):
            obs = pymap.Observation(100, 200, 0.5, 255, 0, 0, 100)
            tm.add_observation(str(shot_id), track_id, obs)
    rec.create_shot(str(n_shots + 5), next(iter(rec.cameras)))
    rec.add_correspondences_from_tracks_manager(tm)
    assert '100' not in rec.points
    for track_id in ['0', '1']:
        pt = rec.points[track_id]
        observations = pt.get_observations()
        assert len(observations) == n_shots