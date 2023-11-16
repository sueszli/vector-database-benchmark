import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from modelscope.models.cv.video_stabilization.utils.image_utils import topk_map
from modelscope.models.cv.video_stabilization.utils.IterativeSmooth import generateSmooth
from modelscope.models.cv.video_stabilization.utils.MedianFilter import MultiMotionPropagate, SingleMotionPropagate
from modelscope.models.cv.video_stabilization.utils.RAFTUtils import InputPadder
from .config import cfg
from .MotionPro import MotionPro
from .RAFT.raft import RAFT
from .rf_det_so import RFDetSO
from .Smoother import Smoother

class KeypointDetction(nn.Module):

    def __init__(self, RFDetPath='', topK=cfg.TRAIN.TOPK, detectorType=0):
        if False:
            while True:
                i = 10
        super(KeypointDetction, self).__init__()
        self.feature_params = dict(maxCorners=topK, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.TOPK = topK
        self.type = detectorType

    def forward(self, im_data):
        if False:
            while True:
                i = 10
        '\n        @param im_data [B, 1, H, W] gray images\n        @return im_topk [B, 1, H, W]\n        @return kpts [[N, 4] for B] (B, 0, H, W)\n        '
        device = im_data.device
        im1 = im_data
        im1 = (im1.cpu().numpy() * 255).astype(np.uint8)
        batch = im1.shape[0]
        assert im1.shape[1] == 1
        im_topK = torch.zeros((batch, 1, im1.shape[2], im1.shape[3]), device=device)
        for idx in range(batch):
            im = im1[idx, 0]
            if self.type == 0:
                p = cv2.goodFeaturesToTrack(im, mask=None, **self.feature_params)
            p = p[:, 0, :]
            im_topK[idx, 0, p[:, 1], p[:, 0]] = 1.0
        kpts = im_topK.nonzero()
        kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(batch)]
        return (im_topK, kpts)

class RFDetection(nn.Module):

    def __init__(self, RFDetPath, topK=cfg.TRAIN.TOPK):
        if False:
            print('Hello World!')
        super(RFDetection, self).__init__()
        self.det = RFDetSO(cfg.TRAIN.score_com_strength, cfg.TRAIN.scale_com_strength, cfg.TRAIN.NMS_THRESH, cfg.TRAIN.NMS_KSIZE, cfg.TRAIN.TOPK, cfg.MODEL.GAUSSIAN_KSIZE, cfg.MODEL.GAUSSIAN_SIGMA, cfg.MODEL.KSIZE, cfg.MODEL.padding, cfg.MODEL.dilation, cfg.MODEL.scale_list)
        self.TOPK = topK

    def forward(self, im_data, batch=2, allInfer=False):
        if False:
            while True:
                i = 10
        '\n        @param im_data [B, 1, H, W]\n        @return im_topk [B, 1, H, W]\n        @return kpts [[N, 4] for B] (B, 0, H, W)\n        '
        if allInfer:
            im_data = im_data
            (im_rawsc, _, _) = self.det(im_data)
            im_score = self.det.process(im_rawsc)[0]
            im_topk = topk_map(im_score, self.TOPK).permute(0, 3, 1, 2)
            kpts = im_topk.nonzero()
            kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(im_data.shape[0])]
            im_topk = im_topk.float()
        else:
            im_topK_ = []
            kpts_ = []
            for j in range(0, im_data.shape[0], batch):
                im_data_clip = im_data[j:j + batch]
                (im_rawsc, _, _) = self.det(im_data_clip)
                im_score = self.det.process(im_rawsc)[0]
                im_topk = topk_map(im_score, self.TOPK).permute(0, 3, 1, 2)
                kpts = im_topk.nonzero()
                kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(im_data_clip.shape[0])]
                im_topk = im_topk.float()
                im_topK_.append(im_topk)
                kpts_ = kpts_ + kpts
            kpts = kpts_
            im_topk = torch.cat(im_topK_, 0)
        return (im_topk, kpts)

    def reload(self, RFDetPath):
        if False:
            for i in range(10):
                print('nop')
        print('reload RFDet Model')
        pretrained_dict = torch.load(RFDetPath)['state_dict']
        model_dict = self.det.state_dict()
        pretrained_dict = {k[4:]: v for (k, v) in pretrained_dict.items() if k[:3] == 'det' and k[4:] in model_dict}
        assert len(pretrained_dict.keys()) > 0
        model_dict.update(pretrained_dict)
        assert len(model_dict.keys()) == len(pretrained_dict.keys()), 'mismatch for RFDet'
        self.det.load_state_dict(model_dict)
        print('successfully load {} params for RFDet'.format(len(model_dict)))

class MotionEstimation(nn.Module):

    def __init__(self, args, RAFTPath=''):
        if False:
            return 10
        super(MotionEstimation, self).__init__()
        self.RAFT = RAFT(args)

    def forward(self, x, x_RGB, im_topk, kpts):
        if False:
            return 10
        '\n        @param im_data [B, 1, H, W]\n        @param im_topk [B, 1, H, W]\n        @param kpts [[N, 4] for B] (B, 0, H, W)\n        @param OpticalFlow [B, 2, H, W] precomputed optical flow; optional, default None\n        @param RGBImages [B, 3, H, W] RGB images for optical flow computation, optional, default None\n        '
        if self.RAFT is None:
            raise NotImplementedError()
        optical_flow = []
        for i in range(0, x_RGB.shape[1] - 1):
            padder = InputPadder(x_RGB[:, i, :, :, :].shape)
            (image1, image2) = padder.pad(x_RGB[:, i, :, :, :], x_RGB[:, i + 1, :, :, :])
            (flow_low, flow_up) = self.RAFT(image1.cuda(), image2.cuda(), iters=20, test_mode=True)
            optical_flow.append(flow_up)
        x_RGB = x_RGB.cpu()
        torch.cuda.empty_cache()
        optical_flow = torch.cat(optical_flow, 0)
        flow_masked = optical_flow * im_topk[:-1]
        return flow_masked

    def reload(self, RAFTPath):
        if False:
            return 10
        self.RAFT.load_state_dict({strKey.replace('module.', ''): tenWeight for (strKey, tenWeight) in torch.load(RAFTPath).items()})
        print('successfully load all params for RAFT')

class KLT(nn.Module):

    def __init__(self, RAFTPath=''):
        if False:
            print('Hello World!')
        super(KLT, self).__init__()
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def forward(self, x, x_RGB, im_topk, kpts):
        if False:
            i = 10
            return i + 15
        '\n        @param im_data [B, 1, H, W]\n        @param im_topk [B, 1, H, W]\n        @param kpts [[N, 4] for B] (B, 0, H, W)\n        @param OpticalFlow [B, 2, H, W] precomputed optical flow; optional, default None\n        @param RGBImages [B, 3, H, W] RGB images for optical flow computation, optional, default None\n        '
        (batch, _, height, width) = x.shape
        im_cpu = (x.cpu().numpy() * 255.0).astype(np.uint8)[:, 0, :, :]
        OpticalFlow = np.zeros((batch - 1, 2, height, width))
        for j in range(batch - 1):
            p0 = kpts[j].detach().cpu().numpy()[:, ::-1]
            p0 = np.expand_dims(p0[:, :2], 1).astype(np.float32)
            (p1, _, _) = cv2.calcOpticalFlowPyrLK(im_cpu[j], im_cpu[j + 1], p0, None, **self.lk_params)
            op = p1 - p0
            p0 = p0.astype(np.uint8)
            OpticalFlow[j, :, p0[:, 0, 1], p0[:, 0, 0]] = op[:, 0, :]
        return torch.from_numpy(OpticalFlow.astype(np.float32)).to(x.device)

class motionPropagate(object):

    def __init__(self, inferenceMethod):
        if False:
            while True:
                i = 10
        self.inference = inferenceMethod

class JacobiSolver(nn.Module):

    def __init__(self):
        if False:
            return 10
        super(JacobiSolver, self).__init__()
        self.generateSmooth = generateSmooth
        self.KernelSmooth = Smoother().KernelSmooth

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return None

class DUT(nn.Module):

    def __init__(self, SmootherPath='', RFDetPath='', RAFTPath='', MotionProPath='', homo=True, args=None):
        if False:
            for i in range(10):
                print('nop')
        super(DUT, self).__init__()
        print('------------------model configuration----------------------')
        if RFDetPath != '':
            print('using RFNet ...')
            self.keypointModule = RFDetection(RFDetPath)
        else:
            print('using corner keypoint detector...')
            self.keypointModule = KeypointDetction()
        if RAFTPath != '':
            print('using RAFT for motion estimation...')
            self.motionEstimation = MotionEstimation(args, RAFTPath)
        else:
            print('using KLT tracker for motion estimation...')
            self.motionEstimation = KLT()
        if MotionProPath != '':
            if homo:
                print('using Motion Propagation model with multi homo...')
                self.motionPro = MotionPro(globalchoice='multi')
            else:
                print('using Motion Propagation model with single homo...')
                self.motionPro = MotionPro(globalchoice='single')
        elif homo:
            print('using median filter with multi homo...')
            self.motionPro = motionPropagate(MultiMotionPropagate)
        else:
            print('using median filter with single homo...')
            self.motionPro = motionPropagate(SingleMotionPropagate)
        if SmootherPath != '':
            print('using Deep Smoother Model...')
            self.smoother = Smoother()
        else:
            print('using Jacobi Solver ...')
            self.smoother = JacobiSolver()
        self.reload(SmootherPath, RFDetPath, RAFTPath, MotionProPath)

    def forward(self, x, x_RGB, repeat=50):
        if False:
            i = 10
            return i + 15
        return self.inference(x, x_RGB, repeat)

    def inference(self, x, x_RGB, repeat=50):
        if False:
            print('Hello World!')
        '\n        @param: x [B, C, T, H, W] Assume B is 1 here, a set of Gray images\n        @param: x_RGB [B, C, T, H, W] Assume B is 1 here, a set of RGB images\n        @param: repeat int repeat time for the smoother module\n\n        @return: smoothPath\n        '
        x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        print('------------------detect keypoints-------------------------')
        (im_topk, kpts) = self.keypointModule.forward(x)
        x = x.cpu()
        torch.cuda.empty_cache()
        print('------------------estimate motion--------------------------')
        masked_flow = self.motionEstimation.forward(x, x_RGB, im_topk, kpts)
        x_RGB = x_RGB.cpu()
        im_topk = im_topk.cpu()
        torch.cuda.empty_cache()
        del x
        del x_RGB
        del im_topk
        print('------------------motion propagation-----------------------')
        origin_motion = [self.motionPro.inference(masked_flow[i:i + 1, 0:1, :, :].cuda(), masked_flow[i:i + 1, 1:2, :, :].cuda(), kpts[i]).cpu() for i in range(len(kpts) - 1)]
        origin_motion = torch.stack(origin_motion, 2).cuda()
        origin_motion = torch.cat([torch.zeros_like(origin_motion[:, :, 0:1, :, :]).to(origin_motion.device), origin_motion], 2)
        origin_motion = torch.cumsum(origin_motion, 2)
        min_value = torch.min(origin_motion)
        origin_motion = origin_motion - min_value
        max_value = torch.max(origin_motion) + 1e-05
        origin_motion = origin_motion / max_value
        smoothKernel = self.smoother(origin_motion.cuda())
        smoothPath = torch.cat(self.smoother.KernelSmooth(smoothKernel, origin_motion.cuda(), repeat), 1)
        smoothPath = smoothPath * max_value + min_value
        origin_motion = origin_motion * max_value + min_value
        return (origin_motion, smoothPath)

    def reload(self, SmootherPath, RFDetPath, RAFTPath, MotionProPath):
        if False:
            for i in range(10):
                print('nop')
        print('------------------reload parameters------------------------')
        if SmootherPath == '':
            print('No parameters for JacobiSolver')
        else:
            print('reload Smoother params')
            pretrained_dict = torch.load(SmootherPath)
            model_dict = self.smoother.state_dict()
            pretrained_dict = {k: v for (k, v) in pretrained_dict.items() if k in model_dict}
            assert len(pretrained_dict.keys()) > 0
            assert len(model_dict.keys()) == len(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            assert len(model_dict.keys()) == len(pretrained_dict.keys())
            self.smoother.load_state_dict(model_dict)
            print('successfully load {} params for smoother'.format(len(model_dict)))
        if RFDetPath != '':
            self.keypointModule.reload(RFDetPath)
        else:
            print('No parameters for Keypoint detector')
        if RAFTPath == '':
            print('No parameters for Optical flow')
        else:
            print('reload RAFT Model')
            self.motionEstimation.reload(RAFTPath)
        if MotionProPath == '':
            print('No parameters for motion propagation')
        else:
            print('reload MotionPropagation Model')
            model_dict_motion = torch.load(MotionProPath)
            model_dict = self.motionPro.state_dict()
            model_dict_motion = {k: v for (k, v) in model_dict_motion.items() if k in model_dict}
            assert len(model_dict_motion.keys()) > 0
            model_dict.update(model_dict_motion)
            assert len(model_dict_motion.keys()) == len(model_dict.keys())
            self.motionPro.load_state_dict(model_dict)
            print('successfully load {} params for MotionPropagation'.format(len(model_dict)))
if __name__ == '__main__':
    im_raw = np.random.randn(1, 3, 20, 240, 320).astype(np.float32)
    im_data = im_raw[:, 0:1, :, :, :]
    im_raw = torch.from_numpy(im_raw).cuda()
    im_data = torch.from_numpy(im_data).cuda()
    model = DUT('1', '2', '3', '4')
    model.cuda()
    model.eval()
    smoothPath = model.inference(im_data, im_raw)