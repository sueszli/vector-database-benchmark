"""
Created by Jaided AI
Released Date: 18/08/2022
Description:
A wrapper for DBNet text detection module for EasyOCR
"""
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from .DBNet.DBNet import DBNet

def test_net(image, detector, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, canvas_size=None, poly=False, device='cpu'):
    if False:
        print('Hello World!')
    '\n    A wrapper for DBNet inference routine.\n\n    Parameters\n    ----------\n    image : np.ndarray or list of np.ndarray\n        OpenCV BGR image array or list of it.\n    detector : obj\n        DBNet text detection object.\n    threshold : float, optional\n        Minimum probability for each pixel of heatmap tensor to be considered\n        as a valid text pixel. The default is 0.2.\n    bbox_min_score : float, optional\n        Minimum score for each detected bounding box to be considered as a\n        valid text bounding box. The default is 0.2.\n    bbox_min_size : int, optional\n        Minimum size for each detected bounding box to be considered as a\n        valid text bounding box. The default is 3.\n    max_candidates : int, optional\n        Maximum number of detected bounding boxes to be considered as \n        candidates for valid text bounding boxes. Setting to 0 implies\n        no maximum. The default is 0.\n    canvas_size : int, optional\n        Target detection size. Input image will be resized such that it\'s \n        shorter side is equal to the closest multiple of 32 to the provided \n        canvas_size. If detection_size is not provided, it will be resized to \n        the closest multiple of 32 each side. If the original size exceeds the \n        min-/max-detection sizes (specified in DBNet_inference.yaml), it will be \n        resized to be within the min-/max-sizes. The default is None.\n    poly : boolean, optional\n        If true, return the bounding boxes as find polygons, otherwise, return\n        as rectagular. The default is False.\n    device : str, optional\n        Device to use. Options are "cpu" and "cuda". The default is \'cpu\'.\n\n    Returns\n    -------\n    bboxes : list of lists\n        List of text bounding boxes in format [left, right, top, bottom].\n    polys : list of lists\n        List of polygon text bounding boxes. If argument poly is set to false,\n        this output will also hold the value of output bboxes\n    '
    if isinstance(image, np.ndarray) and len(image.shape) == 4:
        image_arrs = image
    else:
        image_arrs = [image]
    (images, original_shapes) = zip(*[detector.resize_image(img, canvas_size) for img in image_arrs])
    images = [np.transpose(detector.normalize_image(n_img), (2, 0, 1)) for n_img in images]
    image_tensor = torch.from_numpy(np.array(images)).to(device)
    with torch.no_grad():
        hmap = detector.image2hmap(image_tensor.to(device))
        (bboxes, _) = detector.hmap2bbox(image_tensor, original_shapes, hmap, text_threshold=threshold, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates, as_polygon=False)
        if poly:
            (polys, _) = detector.hmap2bbox(image_tensor, original_shapes, hmap, text_threshold=threshold, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates, as_polygon=True)
        else:
            polys = bboxes
    return (bboxes, polys)

def get_detector(trained_model, backbone='resnet18', device='cpu', quantize=True, cudnn_benchmark=False):
    if False:
        return 10
    '\n    A wrapper to initialize DBNet text detection model\n\n    Parameters\n    ----------\n    trained_model : str\n        Path to trained weight to use.\n    backbone : str\n        Backbone to use. Options are \'resnet18\' or \'resnet50\'. The default is \'resnet18\'.\n    device : str, optional\n        Device to use. Options are "cpu" and "cuda". The default is \'cpu\'.\n    quantize : boolean, optional\n        If use, apply model quantization method to the model. The default is True.\n    cudnn_benchmark : boolen, optional\n        DESCRIPTION. The default is False.\n\n    Returns\n    -------\n    dbnet : obj\n        DBNet text detection object.\n    '
    dbnet = DBNet(initialize_model=False, dynamic_import_relative_path=os.path.join('easyocr', 'DBNet'), device=device, verbose=0)
    if backbone not in ['resnet18', 'resnet50']:
        raise ValueError("Invalid backbone. Options are 'resnet18' or 'resnet50'.")
    dbnet.initialize_model(dbnet.configs[backbone]['model'], trained_model)
    if torch.device(device).type == 'cpu':
        if quantize:
            try:
                torch.quantization.quantize_dynamic(dbnet, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        dbnet.model = torch.nn.DataParallel(dbnet.model).to(device)
        cudnn.benchmark = cudnn_benchmark
    dbnet.model.eval()
    return dbnet

def get_textbox(detector, image, canvas_size=None, poly=False, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, device='cpu', **kwargs):
    if False:
        while True:
            i = 10
    '\n    A compatibility wrapper to allow supporting calling this method while \n    providing argument for other detector classes and reformat output accordingly.\n\n    Parameters\n    ----------\n    detector : obj\n        DBNet text detection object.\n    image : np.ndarray or list of np.ndarray\n        OpenCV BGR image array or list of it.\n    canvas_size : int, optional\n        Target detection size. Please see docstring under method resize_image()\n        for explanation. The default is None.\n    poly : boolean, optional\n        If true, return the bounding boxes as find polygons, otherwise, return\n        as rectagular. The default is False.\n    threshold : float, optional\n        Minimum probability for each pixel of heatmap tensor to be considered\n        as a valid text pixel. The default is 0.2.\n    bbox_min_score : float, optional\n        Minimum score for each detected bounding box to be considered as a\n        valid text bounding box. The default is 0.2.\n    bbox_min_size : int, optional\n        Minimum size for each detected bounding box to be considered as a\n        valid text bounding box. The default is 3.\n    max_candidates : int, optional\n        Maximum number of detected bounding boxes to be considered as \n        candidates for valid text bounding box. Setting it to 0 implies\n        no maximum. The default is 0.\n    device : str, optional\n        Device to use. Options are "cpu" and "cuda". The default is \'cpu\'.\n    **kwargs : keyword arguments\n        Unused. Added to support calling this method while providing argument \n        for other detector class.\n\n    Returns\n    -------\n    result : list of lists\n        List of text bounding boxes in format [left, right, top, bottom].\n    '
    if torch.device(device).type != detector.device:
        raise RuntimeError(' '.join(['DBNet detector is initialized with {} device, but detection routine', 'is called with device = {}.', 'To use this detector both have to be the same.']).format(detector.device, device))
    (_, polys_list) = test_net(image, detector, threshold=threshold, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates, canvas_size=canvas_size, poly=poly, device=device)
    result = [[np.array(box).astype(np.int32).reshape(-1) for box in polys] for polys in polys_list]
    return result