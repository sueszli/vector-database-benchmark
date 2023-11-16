import numpy as np
from .iou_evaluator import DetectionIoUEvaluator

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reset()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return

class QuadMeasurer:

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        if False:
            while True:
                i = 10
        '\n        batch: (image, polygons, ignore_tags\n        batch: a dict produced by dataloaders.\n            image: tensor of shape (N, C, H, W).\n            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.\n            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.\n            shape: the original shape of images.\n            filename: the original filenames of images.\n        output: (polygons, ...)\n        '
        results = []
        gt_polyons_batch = batch['polygons']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for (polygons, pred_polygons, pred_scores, ignore_tags) in zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=polygons[i], ignore=ignore_tags[i]) for i in range(len(polygons))]
            if is_output_polygon:
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        pred.append(dict(points=pred_polygons.reshape(-1, 4, 2)[i, :, :].tolist()))
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        if False:
            for i in range(10):
                print('nop')
        return self.measure(batch, output, is_output_polygon, box_thresh)

    def evaluate_measure(self, batch, output):
        if False:
            i = 10
            return i + 15
        return (self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist())

    def gather_measure(self, raw_metrics):
        if False:
            for i in range(10):
                print('nop')
        raw_metrics = [image_metrics for batch_metrics in raw_metrics for image_metrics in batch_metrics]
        result = self.evaluator.combine_results(raw_metrics)
        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()
        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-08)
        fmeasure.update(fmeasure_score)
        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}