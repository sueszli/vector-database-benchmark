import logging
import numpy as np
import os

class BevParams(object):
    """

    """
    bev_size = None
    bev_res = None
    bev_xLimits = None
    bev_zLimits = None
    imSize = None
    imSize_back = None

    def __init__(self, bev_res, bev_xLimits, bev_zLimits, imSize):
        if False:
            while True:
                i = 10
        '\n\n        @param bev_size:\n        @param bev_res:\n        @param bev_xLimits:\n        @param bev_zLimits:\n        @param imSize:\n        '
        bev_size = (round((bev_zLimits[1] - bev_zLimits[0]) / bev_res), round((bev_xLimits[1] - bev_xLimits[0]) / bev_res))
        self.bev_size = bev_size
        self.bev_res = bev_res
        self.bev_xLimits = bev_xLimits
        self.bev_zLimits = bev_zLimits
        self.imSize = imSize

    def px2meter(self, px_in):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        @param px_in:\n        '
        return px_in * self.bev_res

    def meter2px(self, meter_in):
        if False:
            return 10
        '\n\n        @param meter_in:\n        '
        return meter_in / self.bev_res

    def convertPositionMetric2Pixel(self, YXpointArrays):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        @param YXpointArrays:\n        '
        allY = YXpointArrays[:, 0]
        allX = YXpointArrays[:, 1]
        allYconverted = self.bev_size[0] - self.meter2px(allY - self.bev_zLimits[0])
        allXconverted = self.meter2px(allX - self.bev_xLimits[0])
        return np.array(np.append(allYconverted.reshape((len(allYconverted), 1)), allXconverted.reshape((len(allXconverted), 1)), axis=1))

    def convertPositionPixel2Metric(self, YXpointArrays):
        if False:
            while True:
                i = 10
        '\n\n        @param YXpointArrays:\n        '
        allY = YXpointArrays[:, 0]
        allX = YXpointArrays[:, 1]
        allYconverted = self.px2meter(self.bev_size[0] - allY) + self.bev_zLimits[0]
        allXconverted = self.px2meter(allX) + self.bev_xLimits[0]
        return np.array(np.append(allYconverted.reshape((len(allYconverted), 1)), allXconverted.reshape((len(allXconverted), 1)), axis=1))

    def convertPositionPixel2Metric2(self, inputTupleY, inputTupleX):
        if False:
            while True:
                i = 10
        '\n\n        @param inputTupleY:\n        @param inputTupleX:\n        '
        return (self.px2meter(self.bev_size[0] - inputTupleY) + self.bev_zLimits[0], self.px2meter(inputTupleX) + self.bev_xLimits[0])

def readKittiCalib(filename, dtype='f8'):
    if False:
        while True:
            i = 10
    '\n    \n    :param filename:\n    :param dtype:\n    '
    outdict = dict()
    output = open(filename, 'rb')
    allcontent = output.readlines()
    output.close()
    for contentRaw in allcontent:
        content = contentRaw.strip()
        if content == '':
            continue
        if content[0] != '#':
            tmp = content.split(':')
            assert len(tmp) == 2, 'wrong file format, only one : per line!'
            var = tmp[0].strip()
            values = np.array(tmp[-1].strip().split(' '), dtype)
            outdict[var] = values
    return outdict

class KittiCalibration(object):
    calib_dir = None
    calib_end = None
    R0_rect = None
    P2 = None
    Tr33 = None
    Tr = None
    Tr_cam_to_road = None

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        '
        pass

    def readFromFile(self, filekey=None, fn=None):
        if False:
            while True:
                i = 10
        '\n\n        @param fn:\n        '
        if filekey != None:
            fn = os.path.join(self.calib_dir, filekey + self.calib_end)
        assert fn != None, 'Problem! fn or filekey must be != None'
        cur_calibStuff_dict = readKittiCalib(fn)
        self.setup(cur_calibStuff_dict)

    def setup(self, dictWithKittiStuff, useRect=False):
        if False:
            i = 10
            return i + 15
        '\n\n        @param dictWithKittiStuff:\n        '
        dtype_str = 'f8'
        self.P2 = np.matrix(dictWithKittiStuff['P2']).reshape((3, 4))
        if useRect:
            R2_1 = self.P2
        else:
            R0_rect_raw = np.array(dictWithKittiStuff['R0_rect']).reshape((3, 3))
            self.R0_rect = np.matrix(np.hstack((np.vstack((R0_rect_raw, np.zeros((1, 3), dtype_str))), np.zeros((4, 1), dtype_str))))
            self.R0_rect[3, 3] = 1.0
            R2_1 = np.dot(self.P2, self.R0_rect)
        Tr_cam_to_road_raw = np.array(dictWithKittiStuff['Tr_cam_to_road']).reshape(3, 4)
        self.Tr_cam_to_road = np.matrix(np.vstack((Tr_cam_to_road_raw, np.zeros((1, 4), dtype_str))))
        self.Tr_cam_to_road[3, 3] = 1.0
        self.Tr = np.dot(R2_1, self.Tr_cam_to_road.I)
        self.Tr33 = self.Tr[:, [0, 2, 3]]

    def get_matrix33(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        '
        assert self.Tr33 != None
        return self.Tr33

class BirdsEyeView(object):
    """

    """
    imSize = None
    bevParams = None
    invalid_value = float('-INFINITY')
    im_u_float = None
    im_v_float = None
    bev_x_ind = None
    bev_z_ind = None

    def __init__(self, bev_res=0.05, bev_xRange_minMax=(-10, 10), bev_zRange_minMax=(6, 46)):
        if False:
            i = 10
            return i + 15
        '\n        \n        :param bev_res:\n        :param bev_xRange_minMax:\n        :param bev_zRange_minMax:\n        '
        self.calib = KittiCalibration()
        bev_res = bev_res
        bev_xRange_minMax = bev_xRange_minMax
        bev_zRange_minMax = bev_zRange_minMax
        self.bevParams = BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, self.imSize)

    def world2image(self, X_world, Y_world, Z_world):
        if False:
            return 10
        '\n\n        @param X_world:\n        @param Y_world:\n        @param Z_world:\n        '
        if not type(Y_world) == np.ndarray:
            Y_world = np.ones_like(Z_world) * Y_world
        y = np.vstack((X_world, Y_world, Z_world, np.ones_like(Z_world)))
        test = self.world2image_uvMat(np.vstack((X_world, Z_world, np.ones_like(Z_world))))
        self.xi1 = test[0, :]
        self.yi1 = test[1, :]
        assert self.imSize != None
        condition = ~((self.yi1 >= 1) & (self.xi1 >= 1) & (self.yi1 <= self.imSize[0]) & (self.xi1 <= self.imSize[1]))
        if isinstance(condition, np.ndarray):
            self.xi1[condition] = self.invalid_value
            self.yi1[condition] = self.invalid_value
        elif condition == True:
            self.xi1 = self.invalid_value
            self.yi1 = self.invalid_value

    def world2image_uvMat(self, uv_mat):
        if False:
            while True:
                i = 10
        '\n\n        @param XYZ_mat: is a 4 or 3 times n matrix\n        '
        if uv_mat.shape[0] == 2:
            if len(uv_mat.shape) == 1:
                uv_mat = uv_mat.reshape(uv_mat.shape + (1,))
            uv_mat = np.vstack((uv_mat, np.ones((1, uv_mat.shape[1]), uv_mat.dtype)))
        result = np.dot(self.Tr33, uv_mat)
        resultB = np.broadcast_arrays(result, result[-1, :])
        return resultB[0] / resultB[1]

    def setup(self, calib_file):
        if False:
            i = 10
            return i + 15
        '\n        \n        :param calib_file:\n        '
        self.calib.readFromFile(fn=calib_file)
        self.set_matrix33(self.calib.get_matrix33())

    def set_matrix33(self, matrix33):
        if False:
            i = 10
            return i + 15
        '\n\n        @param matrix33:\n        '
        self.Tr33 = matrix33

    def compute(self, data):
        if False:
            return 10
        '\n        Compute BEV\n        :param data:\n        '
        self.imSize = data.shape
        self.computeBEVLookUpTable()
        return self.transformImage2BEV(data, out_dtype=data.dtype)

    def compute_reverse(self, data, imSize):
        if False:
            while True:
                i = 10
        '\n        Compute BEV\n        :param data:\n        '
        self.imSize = imSize
        self.computeBEVLookUpTable_reverse()
        return self.transformBEV2Image(data, out_dtype=data.dtype)

    def computeBEVLookUpTable_reverse(self, imSize=None):
        if False:
            while True:
                i = 10
        '\n\n        '
        mgrid = np.lib.index_tricks.nd_grid()
        if imSize == None:
            imSize = self.imSize
        self.imSize_back = (imSize[0], imSize[1])
        yx_im = mgrid[1:self.imSize_back[0] + 1, 1:self.imSize_back[1] + 1].astype('i4')
        y_im = yx_im[0, :, :]
        x_im = yx_im[1, :, :]
        dim = self.imSize_back[0] * self.imSize_back[1]
        uvMat = np.vstack((x_im.flatten(), y_im.flatten(), np.ones((dim,), 'f4')))
        xzMat = self.image2world_uvMat(uvMat)
        X = xzMat[0, :].reshape(x_im.shape)
        Z = xzMat[1, :].reshape(x_im.shape)
        XBevInd_reverse_all = np.round((X - self.bevParams.bev_xLimits[0]) / self.bevParams.bev_res).astype('i4')
        ZBevInd_reverse_all = np.round(self.bevParams.bev_size[0] - (Z - self.bevParams.bev_zLimits[0]) / self.bevParams.bev_res).astype('i4')
        self.validMapIm_reverse = (XBevInd_reverse_all >= 1) & (XBevInd_reverse_all <= self.bevParams.bev_size[1]) & (ZBevInd_reverse_all >= 1) & (ZBevInd_reverse_all <= self.bevParams.bev_size[0])
        self.XBevInd_reverse = XBevInd_reverse_all[self.validMapIm_reverse] - 1
        self.ZBevInd_reverse = ZBevInd_reverse_all[self.validMapIm_reverse] - 1
        self.xImInd_reverse = x_im[self.validMapIm_reverse] - 1
        self.yImInd_reverse = y_im[self.validMapIm_reverse] - 1

    def image2world_uvMat(self, uv_mat):
        if False:
            return 10
        '\n\n        @param XYZ_mat: is a 4 or 3 times n matrix\n        '
        if uv_mat.shape[0] == 2:
            if len(uv_mat.shape) == 1:
                uv_mat = uv_mat.reshape(uv_mat.shape + (1,))
            uv_mat = np.vstack((uv_mat, np.ones((1, uv_mat.shape[1]), uv_mat.dtype)))
        result = np.dot(self.Tr33.I, uv_mat)
        resultB = np.broadcast_arrays(result, result[-1, :])
        return resultB[0] / resultB[1]

    def computeBEVLookUpTable(self, cropping_ul=None, cropping_size=None):
        if False:
            while True:
                i = 10
        '\n\n        @param cropping_ul:\n        @param cropping_size:\n        '
        mgrid = np.lib.index_tricks.nd_grid()
        res = self.bevParams.bev_res
        x_vec = np.arange(self.bevParams.bev_xLimits[0] + res / 2, self.bevParams.bev_xLimits[1], res)
        z_vec = np.arange(self.bevParams.bev_zLimits[1] - res / 2, self.bevParams.bev_zLimits[0], -res)
        XZ_mesh = np.meshgrid(x_vec, z_vec)
        assert XZ_mesh[0].shape == self.bevParams.bev_size
        Z_mesh_vec = np.reshape(XZ_mesh[1], self.bevParams.bev_size[0] * self.bevParams.bev_size[1], order='F').astype('f4')
        X_mesh_vec = np.reshape(XZ_mesh[0], self.bevParams.bev_size[0] * self.bevParams.bev_size[1], order='F').astype('f4')
        self.world2image(X_mesh_vec, 0, Z_mesh_vec)
        if cropping_ul is not None:
            valid_selector = np.ones((self.bevParams.bev_size[0] * self.bevParams.bev_size[1],), dtype='bool')
            valid_selector = valid_selector & (self.yi1 >= cropping_ul[0]) & (self.xi1 >= cropping_ul[1])
            if cropping_size is not None:
                valid_selector = valid_selector & (self.yi1 <= cropping_ul[0] + cropping_size[0]) & (self.xi1 <= cropping_ul[1] + cropping_size[1])
            selector = (~(self.xi1 == self.invalid_value)).reshape(valid_selector.shape) & valid_selector
        else:
            selector = ~(self.xi1 == self.invalid_value)
        y_OI_im_sel = self.yi1[selector]
        x_OI_im_sel = self.xi1[selector]
        ZX_ind = mgrid[1:self.bevParams.bev_size[0] + 1, 1:self.bevParams.bev_size[1] + 1].astype('i4')
        Z_ind_vec = np.reshape(ZX_ind[0], selector.shape, order='F')
        X_ind_vec = np.reshape(ZX_ind[1], selector.shape, order='F')
        Z_ind_vec_sel = Z_ind_vec[selector]
        X_ind_vec_sel = X_ind_vec[selector]
        self.im_u_float = x_OI_im_sel
        self.im_v_float = y_OI_im_sel
        self.bev_x_ind = X_ind_vec_sel.reshape(x_OI_im_sel.shape)
        self.bev_z_ind = Z_ind_vec_sel.reshape(y_OI_im_sel.shape)

    def transformImage2BEV(self, inImage, out_dtype='f4'):
        if False:
            for i in range(10):
                print('nop')
        '\n        \n        :param inImage:\n        '
        assert self.im_u_float != None
        assert self.im_v_float != None
        assert self.bev_x_ind != None
        assert self.bev_z_ind != None
        if len(inImage.shape) > 2:
            outputData = np.zeros(self.bevParams.bev_size + (inImage.shape[2],), dtype=out_dtype)
            for channel in xrange(0, inImage.shape[2]):
                outputData[self.bev_z_ind - 1, self.bev_x_ind - 1, channel] = inImage[self.im_v_float.astype('u4') - 1, self.im_u_float.astype('u4') - 1, channel]
        else:
            outputData = np.zeros(self.bevParams.bev_size, dtype=out_dtype)
            outputData[self.bev_z_ind - 1, self.bev_x_ind - 1] = inImage[self.im_v_float.astype('u4') - 1, self.im_u_float.astype('u4') - 1]
        return outputData

    def transformBEV2Image(self, bevMask, out_dtype='f4'):
        if False:
            while True:
                i = 10
        '\n\n        @param bevMask:\n        '
        assert self.xImInd_reverse != None
        assert self.yImInd_reverse != None
        assert self.XBevInd_reverse != None
        assert self.ZBevInd_reverse != None
        assert self.imSize_back != None
        if len(bevMask.shape) > 2:
            outputData = np.zeros(self.imSize_back + (bevMask.shape[2],), dtype=out_dtype)
            for channel in xrange(0, bevMask.shape[2]):
                outputData[self.yImInd_reverse, self.xImInd_reverse, channel] = bevMask[self.ZBevInd_reverse, self.XBevInd_reverse, channel]
        else:
            outputData = np.zeros(self.imSize_back, dtype=out_dtype)
            outputData[self.yImInd_reverse, self.xImInd_reverse] = bevMask[self.ZBevInd_reverse, self.XBevInd_reverse]
        return outputData