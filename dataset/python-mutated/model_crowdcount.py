import os
import time
import base64
import urllib
from io import BytesIO
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from PIL import Image
import tensorflow as tf
import logging
import sys
from crowdcountmcnn.src import network
from crowdcountmcnn.src.crowd_count import CrowdCounter
from abc import ABC, abstractmethod

class CrowdCounting(ABC):

    @abstractmethod
    def score(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

class Router(CrowdCounting):
    """Router model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. 
    """

    def __init__(self, gpu_id=0, mcnn_model_path='mcnn_shtechA_660.h5', cutoff_pose=20, cutoff_mcnn=50):
        if False:
            print('Hello World!')
        self._model_openpose = CrowdCountModelPose(gpu_id)
        self._model_mcnn = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model_path)
        self._cutoff_pose = cutoff_pose
        self._cutoff_mcnn = cutoff_mcnn
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        if False:
            print('Hello World!')
        dict_openpose = self._model_openpose.score(filebytes, return_image, img_dim=img_dim)
        result_openpose = dict_openpose['pred']
        dict_mcnn = self._model_mcnn.score(filebytes, return_image, img_dim=img_dim)
        result_mcnn = dict_mcnn['pred']
        self._logger.info('OpenPose results: {}'.format(result_openpose))
        self._logger.info('MCNN results: {}'.format(result_mcnn))
        if result_openpose > self._cutoff_pose and result_mcnn > self._cutoff_mcnn:
            return dict_mcnn
        else:
            return dict_openpose

class CrowdCountModelMCNN(CrowdCounting):
    """MCNN model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. 
    """

    def __init__(self, gpu_id=0, model_path='mcnn_shtechA_660.h5'):
        if False:
            i = 10
            return i + 15
        self._net = CrowdCounter()
        network.load_net(model_path, self._net)
        if gpu_id == -1:
            self._net.cpu()
        else:
            self._net.cuda(gpu_id)
        self._net.eval()
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        if False:
            i = 10
            return i + 15
        'Score an image. \n        \n        Args:\n            filebytes: Image in stream.\n            return_image (optional): Whether a scored image needs to be returned, defaults to False. \n            img_dim (optional): Max dimension of image, defaults to 1750.\n        \n        Returns:\n            A dictionary with number of people in image, timing for steps, and optionally, returned image.\n        '
        self._logger.info('---started scoring image using MCNN---')
        t = time.time()
        image = load_jpg(filebytes, img_dim)
        t_image_prepare = round(time.time() - t, 3)
        self._logger.info('time on preparing image: {} seconds'.format(t_image_prepare))
        t = time.time()
        (pred_mcnn, model_output) = score_mcnn(self._net, image)
        t_score = round(time.time() - t, 3)
        self._logger.info('time on scoring image: {} seconds'.format(t_score))
        result = {}
        result['pred'] = int(round(pred_mcnn, 0))
        if not return_image:
            dict_time = dict(zip(['t_image_prepare', 't_score'], [t_image_prepare, t_score]))
        else:
            t = time.time()
            scored_image = draw_image_mcnn(model_output)
            t_image_draw = round(time.time() - t, 3)
            self._logger.info('time on drawing image: {}'.format(t_image_draw))
            t = time.time()
            scored_image = web_encode_image(scored_image)
            t_image_encode = round(time.time() - t, 3)
            self._logger.info('time on encoding image: {}'.format(t_image_encode))
            dict_time = dict(zip(['t_image_prepare', 't_score', 't_image_draw', 't_image_encode'], [t_image_prepare, t_score, t_image_draw, t_image_encode]))
            result['image'] = scored_image
        t_total = 0
        for k in dict_time:
            t_total += dict_time[k]
        dict_time['t_total'] = round(t_total, 3)
        self._logger.info('total time: {}'.format(round(t_total, 3)))
        result['time'] = dict_time
        self._logger.info('---finished scoring image---')
        return result

class CrowdCountModelPose(CrowdCounting):
    """OpenPose model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. Set it to -1 to use CPU.
    """

    def __init__(self, gpu_id=0):
        if False:
            print('Hello World!')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        model = 'cmu'
        resize = '656x368'
        (self._w, self._h) = model_wh(resize)
        self._model = init_model(gpu_id, model, self._w, self._h, config)
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        if False:
            while True:
                i = 10
        'Score an image. \n        \n        Args:\n            filebytes: Image in stream.\n            return_image (optional): Whether a scored image needs to be returned, defaults to False. \n            img_dim (optional): Max dimension of image, defaults to 1750.\n        \n        Returns:\n            A dictionary with number of people in image, timing for steps, and optionally, returned image.\n        '
        self._logger.info('---started scoring image using OpenPose---')
        t = time.time()
        img = create_openpose_image(filebytes, img_dim)
        t_image_prepare = round(time.time() - t, 3)
        self._logger.info('time on preparing image: {} seconds'.format(t_image_prepare))
        t = time.time()
        humans = score_openpose(self._model, img, self._w, self._h)
        t_score = round(time.time() - t, 3)
        self._logger.info('time on scoring image: {} seconds'.format(t_score))
        result = {}
        result['pred'] = len(humans)
        if not return_image:
            dict_time = dict(zip(['t_image_prepare', 't_score'], [t_image_prepare, t_score]))
        else:
            t = time.time()
            scored_image = draw_image(img, humans)
            t_image_draw = round(time.time() - t, 3)
            self._logger.info('time on drawing image: {}'.format(t_image_draw))
            t = time.time()
            scored_image = web_encode_image(scored_image)
            t_image_encode = round(time.time() - t, 3)
            self._logger.info('time on encoding image: {}'.format(t_image_encode))
            dict_time = dict(zip(['t_image_prepare', 't_score', 't_image_draw', 't_image_encode'], [t_image_prepare, t_score, t_image_draw, t_image_encode]))
            result['image'] = scored_image
        t_total = 0
        for k in dict_time:
            t_total += dict_time[k]
        dict_time['t_total'] = round(t_total, 3)
        self._logger.info('total time: {}'.format(round(t_total, 3)))
        result['time'] = dict_time
        self._logger.info('---finished scoring image---')
        return result

def init_model(gpu_id, model, w, h, config):
    if False:
        while True:
            i = 10
    'Initialize model.\n    \n    Args:\n        gpu_id: GPU ID. \n    \n    Returns:\n        A TensorFlow model object.\n    '
    if gpu_id == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), tf_config=config)
    else:
        with tf.device('/device:GPU:{}'.format(gpu_id)):
            e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), tf_config=config)
    return e

def create_openpose_image(filebytes, img_dim):
    if False:
        i = 10
        return i + 15
    'Create image from file bytes.\n    \n    Args:\n        filebytes: Image in stream.\n        img_dim: Max dimension of image.\n    \n    Returns:\n        Image in CV2 format. \n    '
    file_bytes = np.fromstring(filebytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    (img, _) = imresizeMaxDim(img, img_dim)
    return img

def load_jpg(file_bytes, img_dim):
    if False:
        while True:
            i = 10
    image = np.fromstring(file_bytes, np.uint8)
    image = cv2.imdecode(image, 0).astype(np.float32)
    (image, _) = imresizeMaxDim(image, img_dim)
    ht = image.shape[0]
    wd = image.shape[1]
    ht_1 = int(ht / 4) * 4
    wd_1 = int(wd / 4) * 4
    image = cv2.resize(image, (wd_1, ht_1))
    image = image.reshape((1, 1, image.shape[0], image.shape[1]))
    return image

def score_openpose(e, image, w, h):
    if False:
        for i in range(10):
            print('nop')
    'Score an image using OpenPose model.\n    \n    Args:\n        e: OpenPose model.\n        image: Image in CV2 format.\n    \n    Returns:\n        Nubmer of people in image.\n    '
    resize_out_ratio = 4.0
    humans = e.inference(image, resize_to_default=w > 0 and h > 0, upsample_size=resize_out_ratio)
    return humans

def draw_image(image, humans):
    if False:
        while True:
            i = 10
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgDebug = Image.fromarray(img)
    return imgDebug

def web_encode_image(scored_image):
    if False:
        print('Hello World!')
    ret_imgio = BytesIO()
    scored_image.save(ret_imgio, 'PNG')
    processed_file = base64.b64encode(ret_imgio.getvalue())
    scored_image = urllib.parse.quote(processed_file)
    return scored_image

def imresizeMaxDim(img, maxDim, boUpscale=False, interpolation=cv2.INTER_CUBIC):
    if False:
        print('Hello World!')
    'Resize image.\n    \n    Args:\n        img: Image in CV2 format. \n        maxDim: Maximum dimension. \n        boUpscale (optional): Defaults to False. \n        interpolation (optional): Defaults to cv2.INTER_CUBIC. \n    \n    Returns:\n        Resized image and scale.\n    '
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1 or boUpscale:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    else:
        scale = 1.0
    return (img, scale)

def score_mcnn(net, image):
    if False:
        while True:
            i = 10
    model_output = net(image)
    model_output_np = model_output.data.cpu().numpy()
    estimated_count = np.sum(model_output_np)
    return (estimated_count, model_output)

def draw_image_mcnn(model_output):
    if False:
        for i in range(10):
            print('nop')
    estimated_density = model_output.data.cpu().numpy()[0, 0, :, :]
    estimated_density = np.uint8(estimated_density * 255 / estimated_density.max())
    im = Image.fromarray(estimated_density, 'L')
    return im