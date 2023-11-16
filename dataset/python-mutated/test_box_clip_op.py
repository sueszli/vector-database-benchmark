import unittest
import numpy as np
from op_test import OpTest

def box_clip(input_box, im_info, output_box):
    if False:
        for i in range(10):
            print('nop')
    im_w = round(im_info[1] / im_info[2])
    im_h = round(im_info[0] / im_info[2])
    output_box[:, :, 0] = np.maximum(np.minimum(input_box[:, :, 0], im_w - 1), 0)
    output_box[:, :, 1] = np.maximum(np.minimum(input_box[:, :, 1], im_h - 1), 0)
    output_box[:, :, 2] = np.maximum(np.minimum(input_box[:, :, 2], im_w - 1), 0)
    output_box[:, :, 3] = np.maximum(np.minimum(input_box[:, :, 3], im_h - 1), 0)

def batch_box_clip(input_boxes, im_info, lod):
    if False:
        while True:
            i = 10
    n = input_boxes.shape[0]
    m = input_boxes.shape[1]
    output_boxes = np.zeros((n, m, 4), dtype=np.float32)
    cur_offset = 0
    for i in range(len(lod)):
        box_clip(input_boxes[cur_offset:cur_offset + lod[i], :, :], im_info[i, :], output_boxes[cur_offset:cur_offset + lod[i], :, :])
        cur_offset += lod[i]
    return output_boxes

class TestBoxClipOp(OpTest):

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'box_clip'
        lod = [[1, 2, 3]]
        input_boxes = np.random.random((6, 10, 4)) * 5
        im_info = np.array([[5, 8, 1.0], [6, 6, 1.0], [7, 5, 1.0]])
        output_boxes = batch_box_clip(input_boxes, im_info, lod[0])
        self.inputs = {'Input': (input_boxes.astype('float32'), lod), 'ImInfo': im_info.astype('float32')}
        self.outputs = {'Output': output_boxes}
if __name__ == '__main__':
    unittest.main()