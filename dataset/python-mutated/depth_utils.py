"""Utilities for processing depth images.
"""
import numpy as np
import src.rotation_utils as ru
import src.utils as utils

def get_camera_matrix(width, height, fov):
    if False:
        print('Hello World!')
    'Returns a camera matrix from image size and fov.'
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = width / 2.0 / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix = utils.Foo(xc=xc, zc=zc, f=f)
    return camera_matrix

def get_point_cloud_from_z(Y, camera_matrix):
    if False:
        i = 10
        return i + 15
    'Projects the depth image Y into a 3D point cloud.\n  Inputs:\n    Y is ...xHxW\n    camera_matrix\n  Outputs:\n    X is positive going right\n    Y is positive into the image\n    Z is positive up in the image\n    XYZ is ...xHxWx3\n  '
    (x, z) = np.meshgrid(np.arange(Y.shape[-1]), np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x - camera_matrix.xc) * Y / camera_matrix.f
    Z = (z - camera_matrix.zc) * Y / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis], Y[..., np.newaxis], Z[..., np.newaxis]), axis=X.ndim)
    return XYZ

def make_geocentric(XYZ, sensor_height, camera_elevation_degree):
    if False:
        while True:
            i = 10
    'Transforms the point cloud into geocentric coordinate frame.\n  Input:\n    XYZ                     : ...x3\n    sensor_height           : height of the sensor\n    camera_elevation_degree : camera elevation to rectify.\n  Output:\n    XYZ : ...x3\n  '
    R = ru.get_r_matrix([1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ

def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    if False:
        print('Hello World!')
    'Bins points into xy-z bins\n  XYZ_cms is ... x H x W x3\n  Outputs is ... x map_size x map_size x (len(z_bins)+1)\n  '
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    map_center = (map_size - 1.0) / 2.0
    counts = []
    isvalids = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution + map_center).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution + map_center).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size, Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)
        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32), minlength=map_size * map_size * n_z_bins)
        count = np.reshape(count, [map_size, map_size, n_z_bins])
        counts.append(count)
        isvalids.append(isvalid)
    counts = np.array(counts).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    isvalids = np.array(isvalids).reshape(list(sh[:-3]) + [sh[-3], sh[-2], 1])
    return (counts, isvalids)