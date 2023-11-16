import os
import cv2
import numpy as np
import tensorflow as tf

def init(mod):
    if False:
        return 10
    PATH_TO_CKPT = mod
    net = tf.Graph()
    with net.as_default():
        od_graph_def = tf.GraphDef()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=net, config=config)
    return (sess, net)

def filter_bboxes_confs(shape, imgsBboxes, imgsConfs, single=False, thresh=0.5):
    if False:
        while True:
            i = 10
    [w, h] = shape
    if single:
        (bboxes, confs) = ([], [])
        for y in range(len(imgsBboxes)):
            if imgsConfs[y] >= thresh:
                [x1, y1, x2, y2] = list(imgsBboxes[y])
                (x1, y1, x2, y2) = (int(w * x1), int(h * y1), int(w * x2), int(h * y2))
                bboxes.append([y1, x1, y2, x2])
                confs.append(imgsConfs[y])
        return (bboxes, confs)
    else:
        (retImgsBboxes, retImgsConfs) = ([], [])
        for x in range(len(imgsBboxes)):
            (bboxes, confs) = ([], [])
            for y in range(len(imgsBboxes[x])):
                if imgsConfs[x][y] >= thresh:
                    [x1, y1, x2, y2] = list(imgsBboxes[x][y])
                    (x1, y1, x2, y2) = (int(w * x1), int(h * y1), int(w * x2), int(h * y2))
                    bboxes.append([y1, x1, y2, x2])
                    confs.append(imgsConfs[x][y])
            retImgsBboxes.append(bboxes)
            retImgsConfs.append(confs)
        return (retImgsBboxes, retImgsConfs)

def detect(im, sess, net):
    if False:
        print('Hello World!')
    image_np = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = net.get_tensor_by_name('image_tensor:0')
    bboxes = net.get_tensor_by_name('detection_boxes:0')
    dConfs = net.get_tensor_by_name('detection_scores:0')
    classes = net.get_tensor_by_name('detection_classes:0')
    num_detections = net.get_tensor_by_name('num_detections:0')
    (bboxes, dConfs, classes, num_detections) = sess.run([bboxes, dConfs, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    (w, h, _) = im.shape
    (bboxes, confs) = filter_bboxes_confs([w, h], bboxes[0], dConfs[0], True)
    return (bboxes, confs)

class FaceDetector:

    def __init__(self, mod):
        if False:
            for i in range(10):
                print('nop')
        (self.sess, self.net) = init(mod)

    def do_detect(self, im):
        if False:
            for i in range(10):
                print('nop')
        (bboxes, confs) = detect(im, self.sess, self.net)
        return (bboxes, confs)