"""
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
"""
from __future__ import print_function
import numpy as np
import cv2
ply_header = 'ply\nformat ascii 1.0\nelement vertex %(vert_num)d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'

def write_ply(fn, verts, colors):
    if False:
        return 10
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.pyrDown(cv2.imread('../data/aloeL.jpg'))
    imgR = cv2.pyrDown(cv2.imread('../data/aloeR.jpg'))
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=16, P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    print('generating 3d point cloud...')
    (h, w) = imgL.shape[:2]
    f = 0.8 * w
    Q = np.float32([[1, 0, 0, -0.5 * w], [0, -1, 0, 0.5 * h], [0, 0, 0, -f], [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()