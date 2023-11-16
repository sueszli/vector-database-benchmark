import argparse
import os
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.tfrecord as tfrec
from test_detection_pipeline import coco_anchors
from test_utils import compare_pipelines, get_dali_extra_path
test_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco')
test_dummy_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco_dummy')

class TFRecordDetectionPipeline(Pipeline):

    def __init__(self, args):
        if False:
            return 10
        super(TFRecordDetectionPipeline, self).__init__(args.batch_size, args.num_workers, 0, 0)
        self.input = ops.readers.TFRecord(path=os.path.join(test_dummy_data_path, 'small_coco.tfrecord'), index_path=os.path.join(test_dummy_data_path, 'small_coco_index.idx'), features={'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ''), 'image/object/class/label': tfrec.VarLenFeature([], tfrec.int64, 0), 'image/object/bbox': tfrec.VarLenFeature([4], tfrec.float32, 0.0)}, shard_id=0, num_shards=1, random_shuffle=False)
        self.decode_gpu = ops.decoders.Image(device='mixed', output_type=types.RGB)
        self.cast = ops.Cast(dtype=types.INT32)
        self.box_encoder = ops.BoxEncoder(device='cpu', criteria=0.5, anchors=coco_anchors())

    def define_graph(self):
        if False:
            print('Hello World!')
        inputs = self.input()
        input_images = inputs['image/encoded']
        image_gpu = self.decode_gpu(input_images)
        labels = self.cast(inputs['image/object/class/label'])
        (encoded_boxes, encoded_labels) = self.box_encoder(inputs['image/object/bbox'], labels)
        return (image_gpu, inputs['image/object/bbox'], labels, encoded_boxes, encoded_labels)

class COCODetectionPipeline(Pipeline):

    def __init__(self, args, data_path=test_data_path):
        if False:
            return 10
        super(COCODetectionPipeline, self).__init__(args.batch_size, args.num_workers, 0, 0)
        self.input = ops.readers.COCO(file_root=os.path.join(data_path, 'images'), annotations_file=os.path.join(data_path, 'instances.json'), shard_id=0, num_shards=1, ratio=True, ltrb=True, random_shuffle=False)
        self.decode_gpu = ops.decoders.Image(device='mixed', output_type=types.RGB)
        self.box_encoder = ops.BoxEncoder(device='cpu', criteria=0.5, anchors=coco_anchors())

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        (inputs, boxes, labels) = self.input(name='Reader')
        image_gpu = self.decode_gpu(inputs)
        (encoded_boxes, encoded_labels) = self.box_encoder(boxes, labels)
        return (image_gpu, boxes, labels, encoded_boxes, encoded_labels)

def print_args(args):
    if False:
        for i in range(10):
            print('nop')
    print('Args values:')
    for arg in vars(args):
        print('{0} = {1}'.format(arg, getattr(args, arg)))
    print()

def run_test(args):
    if False:
        for i in range(10):
            print('nop')
    print_args(args)
    pipe_tf = TFRecordDetectionPipeline(args)
    pipe_coco = COCODetectionPipeline(args, test_dummy_data_path)
    compare_pipelines(pipe_tf, pipe_coco, 1, 64)

def make_parser():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='COCO Tfrecord test')
    parser.add_argument('-i', '--iters', default=None, type=int, metavar='N', help='number of iterations to run (default: whole dataset)')
    parser.add_argument('-w', '--num_workers', default=4, type=int, metavar='N', help='number of worker threads (default: %(default)s)')
    return parser
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.batch_size = 1
    run_test(args)