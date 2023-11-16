import easydict
from utils.nms_wrapper import apply_nms_to_single_image_results

def train_object_detector(cfg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Trains an object detector as specified in the configuration\n    :param cfg: the configuration\n    :return: the eval model of the trained detector\n    '
    detector_name = _get_detector_name(cfg)
    eval_model = None
    print('training {}'.format(detector_name))
    if detector_name == 'FastRCNN':
        from FastRCNN.FastRCNN_train import prepare, train_fast_rcnn
        prepare(cfg, use_arg_parser=False)
        eval_model = train_fast_rcnn(cfg)
    elif detector_name == 'FasterRCNN':
        from FasterRCNN.FasterRCNN_train import prepare, train_faster_rcnn
        prepare(cfg, use_arg_parser=False)
        eval_model = train_faster_rcnn(cfg)
    else:
        print('Unknown detector: {}'.format(detector_name))
    return eval_model

def evaluate_test_set(model, cfg):
    if False:
        while True:
            i = 10
    '\n    Evaluates the given model on the test set as specified in the configuration\n    :param model: the model\n    :param cfg: the configuration\n    :return: AP (average precision) per class\n    '
    detector_name = _get_detector_name(cfg)
    aps = None
    print('evaluating {}'.format(detector_name))
    if detector_name == 'FastRCNN':
        from FastRCNN.FastRCNN_eval import compute_test_set_aps
        aps = compute_test_set_aps(model, cfg)
    elif detector_name == 'FasterRCNN':
        from FasterRCNN.FasterRCNN_eval import compute_test_set_aps
        aps = compute_test_set_aps(model, cfg)
    else:
        print('Unknown detector: {}'.format(detector_name))
    return aps

def evaluate_single_image(model, img_path, cfg):
    if False:
        return 10
    '\n    Computes detection results for the given model on the provided image\n    :param model: the model\n    :param img_path: the path to the image\n    :param cfg: the configuration\n    :return:\n        regressed_rois - the predicted bounding boxes\n        cls_probs - class probabilities per bounding box\n    '
    detector_name = _get_detector_name(cfg)
    regressed_rois = None
    cls_probs = None
    print('detecting objects in image {}'.format(img_path))
    if detector_name == 'FastRCNN':
        from FastRCNN.FastRCNN_eval import FastRCNN_Evaluator
        evaluator = FastRCNN_Evaluator(model, cfg)
        (regressed_rois, cls_probs) = evaluator.process_image(img_path)
    elif detector_name == 'FasterRCNN':
        from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
        evaluator = FasterRCNN_Evaluator(model, cfg)
        (regressed_rois, cls_probs) = evaluator.process_image(img_path)
    else:
        print('Unknown detector: {}'.format(detector_name))
    return (regressed_rois, cls_probs)

def filter_results(regressed_rois, cls_probs, cfg):
    if False:
        i = 10
        return i + 15
    '\n    Filters the provided results by performing NMS (non maximum suppression)\n    :param regressed_rois: the predicted bounding boxes\n    :param cls_probs: class probabilities per bounding box\n    :param cfg: the configuration\n    :return:\n        bboxes - the filtered list of bounding boxes\n        labels - the single class label per bounding box\n        scores - the probability for the assigned class label per bounding box\n    '
    labels = cls_probs.argmax(axis=1)
    scores = cls_probs.max(axis=1)
    nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores, use_gpu_nms=cfg.USE_GPU_NMS, device_id=cfg.GPU_ID, nms_threshold=cfg.RESULTS_NMS_THRESHOLD, conf_threshold=cfg.RESULTS_NMS_CONF_THRESHOLD)
    filtered_bboxes = regressed_rois[nmsKeepIndices]
    filtered_labels = labels[nmsKeepIndices]
    filtered_scores = scores[nmsKeepIndices]
    return (filtered_bboxes, filtered_labels, filtered_scores)

def visualize_results(img_path, bboxes, labels, scores, cfg, store_to_path=None):
    if False:
        i = 10
        return i + 15
    "\n    Renders the detection results (bboxes and labels) onto the image.\n    :param img_path: the path to the image\n    :param bboxes: the predicted bounding boxes\n    :param labels: the single class label per bounding box\n    :param scores: the probability for the assigned class label per bounding box\n    :param cfg: the configuration\n    :param store_to_path: optional: a path where to store the rendered image.\n                          If set to 'None' the image will be displayed on screen.\n    :return:\n    "
    from matplotlib.pyplot import imsave, imshow, show
    from utils.plot_helpers import visualize_detections
    img = visualize_detections(img_path, bboxes, labels, scores, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, classes=cfg['DATA'].CLASSES, draw_negative_rois=cfg.DRAW_NEGATIVE_ROIS)
    if store_to_path is not None:
        imsave(store_to_path, img)
    else:
        imshow(img)
        show()

def _get_detector_name(cfg):
    if False:
        print('Hello World!')
    try:
        detector = cfg['DETECTOR']
    except:
        print("Please specify a 'DETECTOR' in your configuration.")
        detector = None
    return detector

def measure_inference_time(model, img_path, cfg, num_repetitions=100):
    if False:
        print('Hello World!')
    '\n    Computes detection results for the given model on the provided image\n    :param model: the model\n    :param img_path: the path to the image\n    :param cfg: the configuration\n    :return:\n        regressed_rois - the predicted bounding boxes\n        cls_probs - class probabilities per bounding box\n    '
    detector_name = _get_detector_name(cfg)
    print('Measuring inference time (seconds per image) as average over {} runs'.format(num_repetitions))
    if detector_name == 'FastRCNN':
        from FastRCNN.FastRCNN_eval import FastRCNN_Evaluator
        evaluator = FastRCNN_Evaluator(model, cfg)
    elif detector_name == 'FasterRCNN':
        from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
        evaluator = FasterRCNN_Evaluator(model, cfg)
    else:
        print('Unknown detector: {}'.format(detector_name))
        return
    from time import time
    start = time()
    for i in range(num_repetitions):
        (_, _) = evaluator.process_image(img_path)
    total = time() - start
    print('seconds per image: {:2f} (total for {} images: {:2f})'.format(total / num_repetitions, num_repetitions, total))