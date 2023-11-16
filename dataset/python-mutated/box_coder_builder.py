"""A function to build an object detection box coder from configuration."""
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import keypoint_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.protos import box_coder_pb2

def build(box_coder_config):
    if False:
        return 10
    'Builds a box coder object based on the box coder config.\n\n  Args:\n    box_coder_config: A box_coder.proto object containing the config for the\n      desired box coder.\n\n  Returns:\n    BoxCoder based on the config.\n\n  Raises:\n    ValueError: On empty box coder proto.\n  '
    if not isinstance(box_coder_config, box_coder_pb2.BoxCoder):
        raise ValueError('box_coder_config not of type box_coder_pb2.BoxCoder.')
    if box_coder_config.WhichOneof('box_coder_oneof') == 'faster_rcnn_box_coder':
        return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[box_coder_config.faster_rcnn_box_coder.y_scale, box_coder_config.faster_rcnn_box_coder.x_scale, box_coder_config.faster_rcnn_box_coder.height_scale, box_coder_config.faster_rcnn_box_coder.width_scale])
    if box_coder_config.WhichOneof('box_coder_oneof') == 'keypoint_box_coder':
        return keypoint_box_coder.KeypointBoxCoder(box_coder_config.keypoint_box_coder.num_keypoints, scale_factors=[box_coder_config.keypoint_box_coder.y_scale, box_coder_config.keypoint_box_coder.x_scale, box_coder_config.keypoint_box_coder.height_scale, box_coder_config.keypoint_box_coder.width_scale])
    if box_coder_config.WhichOneof('box_coder_oneof') == 'mean_stddev_box_coder':
        return mean_stddev_box_coder.MeanStddevBoxCoder(stddev=box_coder_config.mean_stddev_box_coder.stddev)
    if box_coder_config.WhichOneof('box_coder_oneof') == 'square_box_coder':
        return square_box_coder.SquareBoxCoder(scale_factors=[box_coder_config.square_box_coder.y_scale, box_coder_config.square_box_coder.x_scale, box_coder_config.square_box_coder.length_scale])
    raise ValueError('Empty box coder.')