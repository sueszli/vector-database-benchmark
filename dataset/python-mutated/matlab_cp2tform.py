import numpy as np
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm

class MatlabCp2tormException(Exception):

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'In File {}:{}'.format(__file__, super.__str__(self))

def tformfwd(trans, uv):
    if False:
        for i in range(10):
            print('nop')
    "\n    Function:\n    ----------\n        apply affine transform 'trans' to uv\n\n    Parameters:\n    ----------\n        @trans: 3x3 np.array\n            transform matrix\n        @uv: Kx2 np.array\n            each row is a pair of coordinates (x, y)\n\n    Returns:\n    ----------\n        @xy: Kx2 np.array\n            each row is a pair of transformed coordinates (x, y)\n    "
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def tforminv(trans, uv):
    if False:
        for i in range(10):
            print('nop')
    "\n    Function:\n    ----------\n        apply the inverse of affine transform 'trans' to uv\n\n    Parameters:\n    ----------\n        @trans: 3x3 np.array\n            transform matrix\n        @uv: Kx2 np.array\n            each row is a pair of coordinates (x, y)\n\n    Returns:\n    ----------\n        @xy: Kx2 np.array\n            each row is a pair of inverse-transformed coordinates (x, y)\n    "
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy

def findNonreflectiveSimilarity(uv, xy, options=None):
    if False:
        while True:
            i = 10
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))
    if rank(X) >= 2 * K:
        (r, _, _, _) = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return (T, Tinv)

def findSimilarity(uv, xy, options=None):
    if False:
        return 10
    options = {'K': 2}
    (trans1, trans1_inv) = findNonreflectiveSimilarity(uv, xy, options)
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    (trans2r, trans2r_inv) = findNonreflectiveSimilarity(uv, xyR, options)
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    trans2 = np.dot(trans2r, TreflectY)
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)
    if norm1 <= norm2:
        return (trans1, trans1_inv)
    else:
        trans2_inv = inv(trans2)
        return (trans2, trans2_inv)

def get_similarity_transform(src_pts, dst_pts, reflective=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Function:\n    ----------\n        Find Similarity Transform Matrix 'trans':\n            u = src_pts[:, 0]\n            v = src_pts[:, 1]\n            x = dst_pts[:, 0]\n            y = dst_pts[:, 1]\n            [x, y, 1] = [u, v, 1] * trans\n\n    Parameters:\n    ----------\n        @src_pts: Kx2 np.array\n            source points, each row is a pair of coordinates (x, y)\n        @dst_pts: Kx2 np.array\n            destination points, each row is a pair of transformed\n            coordinates (x, y)\n        @reflective: True or False\n            if True:\n                use reflective similarity transform\n            else:\n                use non-reflective similarity transform\n\n    Returns:\n    ----------\n       @trans: 3x3 np.array\n            transform matrix from uv to xy\n        trans_inv: 3x3 np.array\n            inverse of trans, transform matrix from xy to uv\n    "
    if reflective:
        (trans, trans_inv) = findSimilarity(src_pts, dst_pts)
    else:
        (trans, trans_inv) = findNonreflectiveSimilarity(src_pts, dst_pts)
    return (trans, trans_inv)

def cvt_tform_mat_for_cv2(trans):
    if False:
        print('Hello World!')
    "\n    Function:\n    ----------\n        Convert Transform Matrix 'trans' into 'cv2_trans' which could be\n        directly used by cv2.warpAffine():\n            u = src_pts[:, 0]\n            v = src_pts[:, 1]\n            x = dst_pts[:, 0]\n            y = dst_pts[:, 1]\n            [x, y].T = cv_trans * [u, v, 1].T\n\n    Parameters:\n    ----------\n        @trans: 3x3 np.array\n            transform matrix from uv to xy\n\n    Returns:\n    ----------\n        @cv2_trans: 2x3 np.array\n            transform matrix from src_pts to dst_pts, could be directly used\n            for cv2.warpAffine()\n    "
    cv2_trans = trans[:, 0:2].T
    return cv2_trans

def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    if False:
        i = 10
        return i + 15
    "\n    Function:\n    ----------\n        Find Similarity Transform Matrix 'cv2_trans' which could be\n        directly used by cv2.warpAffine():\n            u = src_pts[:, 0]\n            v = src_pts[:, 1]\n            x = dst_pts[:, 0]\n            y = dst_pts[:, 1]\n            [x, y].T = cv_trans * [u, v, 1].T\n\n    Parameters:\n    ----------\n        @src_pts: Kx2 np.array\n            source points, each row is a pair of coordinates (x, y)\n        @dst_pts: Kx2 np.array\n            destination points, each row is a pair of transformed\n            coordinates (x, y)\n        reflective: True or False\n            if True:\n                use reflective similarity transform\n            else:\n                use non-reflective similarity transform\n\n    Returns:\n    ----------\n        @cv2_trans: 2x3 np.array\n            transform matrix from src_pts to dst_pts, could be directly used\n            for cv2.warpAffine()\n    "
    (trans, trans_inv) = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    cv2_trans_inv = cvt_tform_mat_for_cv2(trans_inv)
    return (cv2_trans, cv2_trans_inv)
if __name__ == '__main__':
    "\n    u = [0, 6, -2]\n    v = [0, 3, 5]\n    x = [-1, 0, 4]\n    y = [-1, -10, 4]\n\n    # In Matlab, run:\n    #\n    #   uv = [u'; v'];\n    #   xy = [x'; y'];\n    #   tform_sim=cp2tform(uv,xy,'similarity');\n    #\n    #   trans = tform_sim.tdata.T\n    #   ans =\n    #       -0.0764   -1.6190         0\n    #        1.6190   -0.0764         0\n    #       -3.2156    0.0290    1.0000\n    #   trans_inv = tform_sim.tdata.Tinv\n    #    ans =\n    #\n    #       -0.0291    0.6163         0\n    #       -0.6163   -0.0291         0\n    #       -0.0756    1.9826    1.0000\n    #    xy_m=tformfwd(tform_sim, u,v)\n    #\n    #    xy_m =\n    #\n    #       -3.2156    0.0290\n    #        1.1833   -9.9143\n    #        5.0323    2.8853\n    #    uv_m=tforminv(tform_sim, x,y)\n    #\n    #    uv_m =\n    #\n    #        0.5698    1.3953\n    #        6.0872    2.2733\n    #       -2.6570    4.3314\n    "
    u = [0, 6, -2]
    v = [0, 3, 5]
    x = [-1, 0, 4]
    y = [-1, -10, 4]
    uv = np.array((u, v)).T
    xy = np.array((x, y)).T
    print('\n--->uv:')
    print(uv)
    print('\n--->xy:')
    print(xy)
    (trans, trans_inv) = get_similarity_transform(uv, xy)
    print('\n--->trans matrix:')
    print(trans)
    print('\n--->trans_inv matrix:')
    print(trans_inv)
    print('\n---> apply transform to uv')
    print('\nxy_m = uv_augmented * trans')
    uv_aug = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy_m = np.dot(uv_aug, trans)
    print(xy_m)
    print('\nxy_m = tformfwd(trans, uv)')
    xy_m = tformfwd(trans, uv)
    print(xy_m)
    print('\n---> apply inverse transform to xy')
    print('\nuv_m = xy_augmented * trans_inv')
    xy_aug = np.hstack((xy, np.ones((xy.shape[0], 1))))
    uv_m = np.dot(xy_aug, trans_inv)
    print(uv_m)
    print('\nuv_m = tformfwd(trans_inv, xy)')
    uv_m = tformfwd(trans_inv, xy)
    print(uv_m)
    uv_m = tforminv(trans, xy)
    print('\nuv_m = tforminv(trans, xy)')
    print(uv_m)