"""
Created on Tue Jan 29 14:46:43 2019

@author: Dcm
"""
'Car detection'
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    if False:
        i = 10
        return i + 15
    '\n    通过阈值来过滤对象和分类的置信度。\n\n    参数：\n        box_confidence  - tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。\n        boxes - tensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。\n        box_class_probs - tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。\n        threshold - 实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。\n\n    返回：\n        scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。\n        boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)\n        classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引\n\n    注意："None"是因为你不知道所选框的确切数量，因为它取决于阈值。\n          比如：如果有10个锚框，scores的实际输出大小将是（10,）\n    '
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    filtering_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return (scores, boxes, classes)

def iou(box1, box2):
    if False:
        print('Hello World!')
    '\n    实现两个锚框的交并比的计算\n\n    参数：\n        box1 - 第一个锚框，元组类型，(x1, y1, x2, y2)\n        box2 - 第二个锚框，元组类型，(x1, y1, x2, y2)\n\n    返回：\n        iou - 实数，交并比。\n    '
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    if False:
        for i in range(10):
            print('nop')
    '\n    为锚框实现非最大值抑制（ Non-max suppression (NMS)）\n\n    参数：\n        scores - tensor类型，维度为(None,)，yolo_filter_boxes()的输出\n        boxes - tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小（见下文）\n        classes - tensor类型，维度为(None,)，yolo_filter_boxes()的输出\n        max_boxes - 整数，预测的锚框数量的最大值\n        iou_threshold - 实数，交并比阈值。\n\n    返回：\n        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值\n        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标\n        classes - tensor类型，维度为(,None)，每个锚框的预测的分类\n\n    注意："None"是明显小于max_boxes的，这个函数也会改变scores、boxes、classes的维度，这会为下一步操作提供方便。\n\n    '
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    return (scores, boxes, classes)

def yolo_eval(yolo_outputs, image_shape=(720.0, 1280.0), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    if False:
        print('Hello World!')
    '\n    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。\n\n    参数：\n        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：\n                        box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)\n                        box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)\n                        box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)\n                        box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)\n        image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)\n        max_boxes - 整数，预测的锚框数量的最大值\n        score_threshold - 实数，可能性阈值。\n        iou_threshold - 实数，交并比阈值。\n\n    返回：\n        scores - tensor类型，维度为(,None)，每个锚框的预测的可能值\n        boxes - tensor类型，维度为(4,None)，预测的锚框的坐标\n        classes - tensor类型，维度为(,None)，每个锚框的预测的分类\n    '
    (box_confidence, box_xy, box_wh, box_class_probs) = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    (scores, boxes, classes) = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    (scores, boxes, classes) = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    return (scores, boxes, classes)
'测试已经训练好了的YOLO模型'
sess = K.get_session()
class_names = read_classes('model_data/coco_classes.txt')
anchors = read_anchors('model_data/yolo_anchors.txt')
image_shape = (720.0, 1280.0)
yolo_model = load_model('model_data/yolo.h5')
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
(scores, boxes, classes) = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    if False:
        while True:
            i = 10
    '\n    运行存储在sess的计算图以预测image_file的边界框，打印出预测的图与信息。\n\n    参数：\n        sess - 包含了YOLO计算图的TensorFlow/Keras的会话。\n        image_file - 存储在images文件夹下的图片名称\n    返回：\n        out_scores - tensor类型，维度为(None,)，锚框的预测的可能值。\n        out_boxes - tensor类型，维度为(None,4)，包含了锚框位置信息。\n        out_classes - tensor类型，维度为(None,)，锚框的预测的分类索引。 \n    '
    (image, image_data) = preprocess_image('images/' + image_file, model_image_size=(608, 608))
    (out_scores, out_boxes, out_classes) = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join('out', image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join('out', image_file))
    imshow(output_image)
    return (out_scores, out_boxes, out_classes)
if __name__ == '__main__':
    (out_scores, out_boxes, out_classes) = predict(sess, 'test.jpg')