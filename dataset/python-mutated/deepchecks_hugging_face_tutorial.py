import torch
from transformers import DetrForObjectDetection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detr_resnet = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
detr_resnet = detr_resnet.to(device)
detr_resnet = detr_resnet.eval()
from typing import Union, List, Iterable
import numpy as np
from deepchecks.vision import VisionData
import torchvision.transforms as T

class COCODETRData:
    """Class for loading the COCO dataset meant for the DETR ResNet50 model`.

    Implement the necessary methods to load the images, labels and generate model predictions in a format comprehensible
     by deepchecks.
    """
    DETR_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        if False:
            print('Hello World!')
        self.transforms = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.label_translation = {}
        detr_shift = 0
        for i in range(len(self.DETR_CLASSES)):
            if self.DETR_CLASSES[i] == 'N/A':
                detr_shift += 1
            self.label_translation[i] = i - detr_shift

    @staticmethod
    def batch_to_labels(batch) -> Union[List[torch.Tensor], torch.Tensor]:
        if False:
            while True:
                i = 10
        'Convert the batch to a list of labels. Copied from deepchecks.vision.datasets.detection.coco'

        def move_class(tensor):
            if False:
                while True:
                    i = 10
            return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) if len(tensor) > 0 else tensor
        return [move_class(tensor) for tensor in batch[1]]

    @staticmethod
    def batch_to_images(batch) -> Iterable[np.ndarray]:
        if False:
            print('Hello World!')
        'Convert the batch to a list of images. Copied from deepchecks.vision.datasets.detection.coco'
        return [np.array(x) for x in batch[0]]

    def _detect(self, im, model, device):
        if False:
            return 10
        'A helper function. Applies DETR detection to a single PIL image.'

        def box_cxcywh_to_xyxy(x):
            if False:
                for i in range(10):
                    print('nop')
            'Convert bounding box format from [cx, cy, w, h] to [xmin, ymin, xmax, ymax], when c is "center".'
            (x_c, y_c, w, h) = x.unbind(1)
            b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
            return torch.stack(b, dim=1).clip(0, 1)

        def rescale_bboxes(out_bbox, size):
            if False:
                while True:
                    i = 10
            "Rescale bounding boxes from the DETR model's normalized output to the original image size."
            (img_w, img_h) = size
            b = box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b
        img = self.transforms(im).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img.to(device))
        probas = outputs['logits'].softmax(-1)[0, :, :-1].cpu()
        keep = probas.max(-1).values > 0.7
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), im.size)
        return (probas[keep], bboxes_scaled)

    def _convert_to_80_labels(self, labels):
        if False:
            for i in range(10):
                print('nop')
        'Use the pre-built self.label_translation to translate the DETR predictions to YOLO COCO classes.'
        return torch.Tensor([self.label_translation[label] for label in labels]).reshape((-1, 1))

    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        if False:
            while True:
                i = 10
        'Infer on a batch of images and return it in deepchecks format.\n\n        Return a list of prediction tensors (one for each image) containing in each row:\n        [x_min, y_min, width, height, confidence, class_id]\n        '
        processed_preds = []
        for batch_idx in range(len(batch[0])):
            (probas, bboxes_scaled) = self._detect(batch[0][batch_idx], model, device)
            bboxes_scaled[:, 2:] = bboxes_scaled[:, 2:] - bboxes_scaled[:, :2]
            if len(probas) > 0:
                processed_pred = torch.cat([bboxes_scaled, probas.max(dim=1)[0].reshape((-1, 1)), self._convert_to_80_labels(probas.argmax(dim=1).tolist())], dim=1)
                processed_preds.append(processed_pred)
        return processed_preds
from deepchecks.vision.datasets.detection import coco_torch as coco
from deepchecks.vision.datasets.detection import coco_utils
from deepchecks.vision.vision_data import BatchOutputFormat
detr_train_datalaoder = coco.load_dataset(batch_size=8, object_type='DataLoader')
detr_test_datalaoder = coco.load_dataset(batch_size=8, train=False, object_type='DataLoader')

def deepchecks_collate_fn_generator(model, device):
    if False:
        i = 10
        return i + 15
    'Generates a collate function that converts the batch to the deepchecks format, using the given model.'
    detr_formatter = COCODETRData()

    def deepchecks_collate_fn(batch):
        if False:
            print('Hello World!')
        'A collate function that converts the batch to the format expected by deepchecks.'
        batch = list(zip(*batch))
        images = detr_formatter.batch_to_images(batch)
        labels = detr_formatter.batch_to_labels(batch)
        predictions = detr_formatter.infer_on_batch(batch, model, device)
        return BatchOutputFormat(images=images, labels=labels, predictions=predictions)
    return deepchecks_collate_fn
detr_test_datalaoder.collate_fn = deepchecks_collate_fn_generator(detr_resnet, device)
detr_test_ds = VisionData(detr_test_datalaoder, task_type='object_detection', label_map=coco_utils.LABEL_MAP)
detr_test_ds.head()
yolo_test_ds = coco.load_dataset(object_type='VisionData', train=False)
from deepchecks.vision.checks import MeanAveragePrecisionReport
yolo_map_result = MeanAveragePrecisionReport().run(yolo_test_ds)
yolo_map_result.show()
detr_map_result = MeanAveragePrecisionReport().run(detr_test_ds)
detr_map_result.show()
yolo_map_result.show()