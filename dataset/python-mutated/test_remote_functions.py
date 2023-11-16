import os
import open3d as o3d
import numpy as np
import pytest
if os.name == 'nt':
    address = 'tcp://127.0.0.1:51455'
else:
    address = 'ipc:///tmp/open3d_ipc'

def test_external_visualizer():
    if False:
        return 10
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    receiver = o3d.io.rpc._DummyReceiver(address=address)
    receiver.start()
    ev = o3d.visualization.ExternalVisualizer(address=address)
    mesh = o3d.geometry.TriangleMesh.create_torus()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(100, 3)))
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = np.eye(4)
    assert ev.set(pcd, path='bla/pcd', time=42)
    assert ev.set(mesh, path='bla/mesh', time=42)
    assert ev.set(camera, path='bla/camera', time=42)
    assert ev.set(obj=[pcd, mesh, camera])
    assert ev.set(obj=[(pcd, 'pcd', 1), (mesh, 'mesh', 2), (camera, 'camera', 3)])
    ev.set_time(10)
    ev.set_active_camera('camera')
    receiver.stop()