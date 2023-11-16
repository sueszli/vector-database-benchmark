"""
Created by Jaided AI
Released Date: 18/08/2022
Description:
DBNet text detection module. 
Many parts of the codes are adapted from https://github.com/MhLiao/DB
"""
import os
import math
import yaml
from shapely.geometry import Polygon
import PIL.Image
import numpy as np
import cv2
import pyclipper
import torch
from .model.constructor import Configurable

class DBNet:

    def __init__(self, backbone='resnet18', weight_dir=None, weight_name='pretrained', initialize_model=True, dynamic_import_relative_path=None, device='cuda', verbose=0):
        if False:
            print('Hello World!')
        '\n        DBNet text detector class\n\n        Parameters\n        ----------\n        backbone : str, optional\n            Backbone to use. Options are "resnet18" and "resnet50". The default is "resnet18".\n        weight_dir : str, optional\n            Path to directory that contains weight files. If set to None, the path will be set\n            to "../weights/". The default is None.\n        weight_name : str, optional\n            Name of the weight to use as specified in DBNet_inference.yaml or a filename \n            in weight_dir. The default is \'pretrained\'.\n        initialize_model : Boolean, optional\n            If True, construct the model and load weight at class initialization.\n            Otherwise, only initial the class without constructing the model.\n            The default is True.\n        dynamic_import_relative_path : str, optional\n            Relative path to \'model/detector.py\'. This option is for supporting\n            integrating this module into other modules. For example, easyocr/DBNet\n            This should be left as None when calling this module as a standalone. \n            The default is None.\n        device : str, optional\n            Device to use. Options are "cuda" and "cpu". The default is \'cuda\'.\n        verbose : int, optional\n            Verbosity level. The default is 0.\n\n        Raises\n        ------\n        ValueError\n            Raised when backbone is invalid.\n        FileNotFoundError\n            Raised when weight file is not found.\n\n        Returns\n        -------\n        None.\n        '
        self.device = device
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'DBNet_inference.yaml')
        with open(config_path, 'r') as fid:
            self.configs = yaml.safe_load(fid)
        if dynamic_import_relative_path is not None:
            self.configs = self.set_relative_import_path(self.configs, dynamic_import_relative_path)
        if backbone in self.configs.keys():
            self.backbone = backbone
        else:
            raise ValueError('Invalid backbone. Current support backbone are {}.'.format(','.join(self.configs.keys())))
        if weight_dir is not None:
            self.weight_dir = weight_dir
        else:
            self.weight_dir = os.path.join(os.path.dirname(__file__), 'weights')
        if initialize_model:
            if weight_name in self.configs[backbone]['weight'].keys():
                weight_path = os.path.join(self.weight_dir, self.configs[backbone]['weight'][weight_name])
                error_message = 'A weight with a name {} is found in DBNet_inference.yaml but cannot be find file: {}.'
            else:
                weight_path = os.path.join(self.weight_dir, weight_name)
                error_message = 'A weight with a name {} is not found in DBNet_inference.yaml and cannot be find file: {}.'
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(error_message.format(weight_name, weight_path))
            self.initialize_model(self.configs[backbone]['model'], weight_path)
        else:
            self.model = None
        self.BGR_MEAN = np.array(self.configs['BGR_MEAN'])
        self.min_detection_size = self.configs['min_detection_size']
        self.max_detection_size = self.configs['max_detection_size']

    def set_relative_import_path(self, configs, dynamic_import_relative_path):
        if False:
            i = 10
            return i + 15
        "\n        Create relative import paths for modules specified in class. This method\n        is recursive.\n\n        Parameters\n        ----------\n        configs : dict\n            Configuration dictionary from .yaml file.\n        dynamic_import_relative_path : str, optional\n            Relative path to 'model/detector/'. This option is for supporting\n            integrating this module into other modules. For example, easyocr/DBNet\n            This should be left as None when calling this module as a standalone. \n            The default is None.\n        \n        Returns\n        -------\n        configs : dict\n            Configuration dictionary with correct relative path.\n        "
        assert dynamic_import_relative_path is not None
        prefices = dynamic_import_relative_path.split(os.sep)
        for (key, value) in configs.items():
            if key == 'class':
                configs.update({key: '.'.join(prefices + value.split('.'))})
            elif isinstance(value, dict):
                value = self.set_relative_import_path(value, dynamic_import_relative_path)
            else:
                pass
        return configs

    def load_weight(self, weight_path):
        if False:
            print('Hello World!')
        '\n        Load weight to model.\n\n        Parameters\n        ----------\n        weight_path : str\n            Path to trained weight.\n\n        Raises\n        ------\n        RuntimeError\n            Raised when the model has not yet been contructed.\n\n        Returns\n        -------\n        None.\n        '
        if self.model is None:
            raise RuntimeError('model has not yet been constructed.')
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False)
        self.model.eval()

    def construct_model(self, config):
        if False:
            return 10
        '\n        Contruct text detection model based on the configuration in .yaml file.\n\n        Parameters\n        ----------\n        config : dict\n            Configuration dictionary.\n\n        Returns\n        -------\n        None.\n        '
        self.model = Configurable.construct_class_from_config(config).structure.builder.build(self.device)

    def initialize_model(self, model_config, weight_path):
        if False:
            i = 10
            return i + 15
        '\n        Wrapper to initialize text detection model. This model includes contructing\n        and weight loading.\n\n        Parameters\n        ----------\n        model_config : dict\n            Configuration dictionary.\n        weight_path : str\n            Path to trained weight.\n\n        Returns\n        -------\n        None.\n        '
        self.construct_model(model_config)
        self.load_weight(weight_path)
        if isinstance(self.model.model, torch.nn.DataParallel) and self.device == 'cpu':
            self.model.model = self.model.model.module.to(self.device)

    def get_cv2_image(self, image):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load or convert input to OpenCV BGR image numpy array.\n\n        Parameters\n        ----------\n        image : str, PIL.Image, or np.ndarray\n            Image to load or convert.\n\n        Raises\n        ------\n        FileNotFoundError\n            Raised when the input is a path to file (str), but the file is not found.\n        TypeError\n            Raised when the data type of the input is not supported.\n\n        Returns\n        -------\n        image : np.ndarray\n            OpenCV BGR image.\n        '
        if isinstance(image, str):
            if os.path.isfile(image):
                image = cv2.imread(image, cv2.IMREAD_COLOR).astype('float32')
            else:
                raise FileNotFoundError('Cannot find {}'.format(image))
        elif isinstance(image, np.ndarray):
            image = image.astype('float32')
        elif isinstance(image, PIL.Image.Image):
            image = np.asarray(image)[:, :, ::-1]
        else:
            raise TypeError('Unsupport image format. Only path-to-file, opencv BGR image, and PIL image are supported.')
        return image

    def resize_image(self, img, detection_size=None):
        if False:
            return 10
        '\n        Resize image such that the shorter side of the image is equal to the \n        closest multiple of 32 to the provided detection_size. If detection_size\n        is not provided, it will be resized to the closest multiple of 32 each\n        side. If the original size exceeds the min-/max-detection sizes \n        (specified in configs.yaml), it will be resized to be within the \n        min-/max-sizes.\n\n        Parameters\n        ----------\n        img : np.ndarray\n            OpenCV BGR image.\n        detection_size : int, optional\n            Target detection size. The default is None.\n\n        Returns\n        -------\n        np.ndarray\n            Resized OpenCV BGR image. The width and height of this image should\n            be multiple of 32.\n        '
        (height, width, _) = img.shape
        if detection_size is None:
            detection_size = max(self.min_detection_size, min(height, width, self.max_detection_size))
        if height < width:
            new_height = int(math.ceil(detection_size / 32) * 32)
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = int(math.ceil(detection_size / 32) * 32)
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return (resized_img, (height, width))

    def image_array2tensor(self, image):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert image array (assuming OpenCV BGR format) to image tensor.\n\n        Parameters\n        ----------\n        image : np.ndarray\n            OpenCV BGR image.\n\n        Returns\n        -------\n        torch.tensor\n            Tensor image with 4 dimension [batch, channel, width, height].\n        '
        return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    def normalize_image(self, image):
        if False:
            print('Hello World!')
        '\n        Normalize image by substracting BGR mean and divided by 255\n\n        Parameters\n        ----------\n        image : np.ndarray\n            OpenCV BGR image.\n\n        Returns\n        -------\n        np.ndarray\n            OpenCV BGR image.\n        '
        return (image - self.BGR_MEAN) / 255.0

    def load_image(self, image_path, detection_size=0):
        if False:
            return 10
        '\n        Wrapper to load and convert an image to an image tensor\n\n        Parameters\n        ----------\n        image : path-to-file, PIL.Image, or np.ndarray\n            Image to load or convert.\n        detection_size : int, optional\n            Target detection size. The default is None.\n\n        Returns\n        -------\n        img : torch.tensor\n            Tensor image with 4 dimension [batch, channel, width, height]..\n        original_shape : tuple\n            A tuple (height, width) of the original input image before resizing.\n        '
        img = self.get_cv2_image(image_path)
        (img, original_shape) = self.resize_image(img, detection_size=detection_size)
        img = self.normalize_image(img)
        img = self.image_array2tensor(img)
        return (img, original_shape)

    def load_images(self, images, detection_size=None):
        if False:
            while True:
                i = 10
        '\n        Wrapper to load or convert list of multiple images to a single image \n        tensor. Multiple images are concatenated together on the first dimension.\n        \n        Parameters\n        ----------\n        images : a list of path-to-file, PIL.Image, or np.ndarray\n            Image to load or convert.\n        detection_size : int, optional\n            Target detection size. The default is None.\n\n        Returns\n        -------\n        img : torch.tensor\n            A single tensor image with 4 dimension [batch, channel, width, height].\n        original_shape : tuple\n            A list of tuples (height, width) of the original input image before resizing.\n        '
        (images, original_shapes) = zip(*[self.load_image(image, detection_size=detection_size) for image in images])
        return (torch.cat(images, dim=0), original_shapes)

    def hmap2bbox(self, image_tensor, original_shapes, hmap, text_threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, as_polygon=False):
        if False:
            return 10
        '\n        Translate probability heatmap tensor to text region boudning boxes.\n\n        Parameters\n        ----------\n        image_tensor : torch.tensor\n            Image tensor.\n        original_shapes : tuple\n            Original size of the image (height, width) of the input image (before\n            rounded to the closest multiple of 32).\n        hmap : torch.tensor\n            Probability heatmap tensor.\n        text_threshold : float, optional\n            Minimum probability for each pixel of heatmap tensor to be considered\n            as a valid text pixel. The default is 0.2.\n        bbox_min_score : float, optional\n            Minimum score for each detected bounding box to be considered as a\n            valid text bounding box. The default is 0.2.\n        bbox_min_size : int, optional\n            Minimum size for each detected bounding box to be considered as a\n            valid text bounding box. The default is 3.\n        max_candidates : int, optional\n            Maximum number of detected bounding boxes to be considered as \n            candidates for valid text bounding box. Setting it to 0 implies\n            no maximum. The default is 0.\n        as_polygon : boolean, optional\n            If True, return the bounding box as polygon (fine vertrices), \n            otherwise return as rectangular. The default is False.\n\n        Returns\n        -------\n        boxes_batch : list of lists\n            Bounding boxes of each text box.\n        scores_batch : list of floats\n            Confidence scores of each text box.\n\n        '
        segmentation = self.binarize(hmap, threshold=text_threshold)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(image_tensor.size(0)):
            (height, width) = original_shapes[batch_index]
            if as_polygon:
                (boxes, scores) = self.polygons_from_bitmap(hmap[batch_index], segmentation[batch_index], width, height, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates)
            else:
                (boxes, scores) = self.boxes_from_bitmap(hmap[batch_index], segmentation[batch_index], width, height, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        (boxes_batch, scores_batch) = zip(*[zip(*[(box, score) for (box, score) in zip(boxes, scores) if score > 0]) if any(scores > 0) else [(), ()] for (boxes, scores) in zip(boxes_batch, scores_batch)])
        return (boxes_batch, scores_batch)

    def binarize(self, tensor, threshold):
        if False:
            return 10
        '\n        Apply threshold to return boolean tensor.\n\n        Parameters\n        ----------\n        tensor : torch.tensor\n            input tensor.\n        threshold : float\n            Threshold.\n\n        Returns\n        -------\n        torch.tensor\n            Boolean tensor.\n\n        '
        return tensor > threshold

    def polygons_from_bitmap(self, hmap, segmentation, dest_width, dest_height, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0):
        if False:
            while True:
                i = 10
        '\n        Translate boolean tensor to fine polygon indicating text bounding boxes\n\n        Parameters\n        ----------\n        hmap : torch.tensor\n            Probability heatmap tensor.\n        segmentation : torch.tensor\n            Segmentataion tensor.\n        dest_width : TYPE\n            target width of the output.\n        dest_height : TYPE\n            target width of the output.\n        bbox_min_score : float, optional\n            Minimum score for each detected bounding box to be considered as a\n            valid text bounding box. The default is 0.2.\n        bbox_min_size : int, optional\n            Minimum size for each detected bounding box to be considered as a\n            valid text bounding box. The default is 3.\n        max_candidates : int, optional\n            Maximum number of detected bounding boxes to be considered as \n            candidates for valid text bounding box. Setting it to 0 implies\n            no maximum. The default is 0.\n        \n        Returns\n        -------\n        boxes_batch : list of lists\n            Polygon bounding boxes of each text box.\n        scores_batch : list of floats\n            Confidence scores of each text box.\n\n        '
        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]
        hmap = hmap.cpu().detach().numpy()[0]
        (height, width) = bitmap.shape
        boxes = []
        scores = []
        (contours, _) = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if max_candidates > 0:
            contours = contours[:max_candidates]
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(hmap, points.reshape(-1, 2))
            if score < bbox_min_score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            (_, sside) = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < bbox_min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return (boxes, scores)

    def boxes_from_bitmap(self, hmap, segmentation, dest_width, dest_height, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0):
        if False:
            i = 10
            return i + 15
        '\n        Translate boolean tensor to fine polygon indicating text bounding boxes\n\n        Parameters\n        ----------\n        hmap : torch.tensor\n            Probability heatmap tensor.\n        segmentation : torch.tensor\n            Segmentataion tensor.\n        dest_width : TYPE\n            target width of the output.\n        dest_height : TYPE\n            target width of the output.\n        bbox_min_score : float, optional\n            Minimum score for each detected bounding box to be considered as a\n            valid text bounding box. The default is 0.2.\n        bbox_min_size : int, optional\n            Minimum size for each detected bounding box to be considered as a\n            valid text bounding box. The default is 3.\n        max_candidates : int, optional\n            Maximum number of detected bounding boxes to be considered as \n            candidates for valid text bounding box. Setting it to 0 implies\n            no maximum. The default is 0.\n        \n        Returns\n        -------\n        boxes_batch : list of lists\n            Polygon bounding boxes of each text box.\n        scores_batch : list of floats\n            Confidence scores of each text box.\n        '
        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]
        hmap = hmap.cpu().detach().numpy()[0]
        (height, width) = bitmap.shape
        (contours, _) = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if max_candidates > 0:
            num_contours = min(len(contours), max_candidates)
        else:
            num_contours = len(contours)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        for index in range(num_contours):
            contour = contours[index]
            (points, sside) = self.get_mini_boxes(contour)
            if sside < bbox_min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(hmap, points.reshape(-1, 2))
            if score < bbox_min_score:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            (box, sside) = self.get_mini_boxes(box)
            if sside < bbox_min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return (boxes.tolist(), scores)

    def unclip(self, box, unclip_ratio=1.5):
        if False:
            print('Hello World!')
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        if False:
            for i in range(10):
                print('nop')
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        (index_1, index_2, index_3, index_4) = (0, 1, 2, 3)
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return (box, min(bounding_box[1]))

    def box_score_fast(self, hmap, box_):
        if False:
            print('Hello World!')
        '\n        Calculate total score of each bounding box\n\n        Parameters\n        ----------\n        hmap : torch.tensor\n            Probability heatmap tensor.\n        box_ : list\n            Rectanguar bounding box.\n\n        Returns\n        -------\n        float\n            Confidence score.\n        '
        (h, w) = hmap.shape[:2]
        box = box_.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(hmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def image2hmap(self, image_tensor):
        if False:
            i = 10
            return i + 15
        '\n        Run the model to obtain a heatmap tensor from a image tensor. The heatmap\n        tensor indicates the probability of each pixel being a part of text area.\n\n        Parameters\n        ----------\n        image_tensor : torch.tensor\n            Image tensor.\n\n        Returns\n        -------\n        torch.tensor\n            Probability heatmap tensor.\n        '
        return self.model.forward(image_tensor, training=False)

    def inference(self, image, text_threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, detection_size=None, as_polygon=False, return_scores=False):
        if False:
            print('Hello World!')
        '\n        Wrapper to run the model on an input image to get text bounding boxes.\n\n        Parameters\n        ----------\n        image : path-to-file, PIL.Image, or np.ndarray\n            Image to load or convert.\n        text_threshold : float, optional\n            Minimum probability for each pixel of heatmap tensor to be considered\n            as a valid text pixel. The default is 0.2.\n        bbox_min_score : float, optional\n            Minimum score for each detected bounding box to be considered as a\n            valid text bounding box. The default is 0.2.\n        bbox_min_size : int, optional\n            Minimum size for each detected bounding box to be considered as a\n            valid text bounding box. The default is 3.\n        max_candidates : int, optional\n            Maximum number of detected bounding boxes to be considered as \n            candidates for valid text bounding box. Setting it to 0 implies\n            no maximum. The default is 0.\n        detection_size : int, optional\n            Target detection size. Please see docstring under method resize_image()\n            for explanation. The default is None.\n        as_polygon : boolean, optional\n            If true, return the bounding boxes as find polygons, otherwise, return\n            as rectagular. The default is False.\n        return_scores : boolean, optional\n            If true, return confidence score along with the text bounding boxes.\n            The default is False.\n\n        Returns\n        -------\n        list of lists\n            Text bounding boxes. If return_scores is set to true, another list\n            of lists will also be returned.\n\n        '
        if not isinstance(image, list):
            image = [image]
        (image_tensor, original_shapes) = self.load_images(image, detection_size=detection_size)
        with torch.no_grad():
            hmap = self.image2hmap(image_tensor)
            (batch_boxes, batch_scores) = self.hmap2bbox(image_tensor, original_shapes, hmap, text_threshold=text_threshold, bbox_min_score=bbox_min_score, bbox_min_size=bbox_min_size, max_candidates=max_candidates, as_polygon=as_polygon)
        if return_scores:
            return (batch_boxes, batch_scores)
        else:
            return batch_boxes