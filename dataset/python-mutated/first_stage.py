import math
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from .box_utils import _preprocess, nms

def run_first_stage(image, net, scale, threshold, device='cuda'):
    if False:
        i = 10
        return i + 15
    "Run P-Net, generate bounding boxes, and do NMS.\n\n    Arguments:\n        image: an instance of PIL.Image.\n        net: an instance of pytorch's nn.Module, P-Net.\n        scale: a float number,\n            scale width and height of the image by this number.\n        threshold: a float number,\n            threshold on the probability of a face when generating\n            bounding boxes from predictions of the net.\n\n    Returns:\n        a float numpy array of shape [n_boxes, 9],\n            bounding boxes with scores and offsets (4 + 1 + 4).\n    "
    (width, height) = image.size
    (sw, sh) = (math.ceil(width * scale), math.ceil(height * scale))
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')
    img = Variable(torch.FloatTensor(_preprocess(img)), volatile=True).to(device)
    output = net(img)
    probs = output[1].cpu().data.numpy()[0, 1, :, :]
    offsets = output[0].cpu().data.numpy()
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None
    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]

def _generate_bboxes(probs, offsets, scale, threshold):
    if False:
        while True:
            i = 10
    'Generate bounding boxes at places\n    where there is probably a face.\n\n    Arguments:\n        probs: a float numpy array of shape [n, m].\n        offsets: a float numpy array of shape [1, 4, n, m].\n        scale: a float number,\n            width and height of the image were scaled by this number.\n        threshold: a float number.\n\n    Returns:\n        a float numpy array of shape [n_boxes, 9]\n    '
    stride = 2
    cell_size = 12
    inds = np.where(probs > threshold)
    if inds[0].size == 0:
        return np.array([])
    (tx1, ty1, tx2, ty2) = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    bounding_boxes = np.vstack([np.round((stride * inds[1] + 1.0) / scale), np.round((stride * inds[0] + 1.0) / scale), np.round((stride * inds[1] + 1.0 + cell_size) / scale), np.round((stride * inds[0] + 1.0 + cell_size) / scale), score, offsets])
    return bounding_boxes.T