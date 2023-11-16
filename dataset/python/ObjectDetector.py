from Model import Model
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import os.path
from os import path
from PIL import Image, ImageFont
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

class ObjectDetector:
    def __init__(self):
        ImageFont.truetype('Arial.ttf', 30)
        self.model = Model.getInstance()
        self.model_setup()
        #self.downloadModel()

    #unused method in current version
    def downloadModel(self):
        # Download Model
        if path.isfile(self.PATH_TO_CKPT) != True:
            opener = urllib.request.URLopener()
            opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
            tar_file = tarfile.open(self.MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())

    def loadModel(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            print(self.PATH_TO_CKPT)
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def setCustomModelSettings(self):
        self.MODEL_NAME = '/home/yellow/models/research/object_detection/ODS/inference_graph'
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'  #
        self.PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')  #
        self.NUM_CLASSES = 5

    def updateName(self):
        self.MODEL_NAME = self.model.get_name()
        self.PATH_TO_CKPT = 'models/'+ self.MODEL_NAME + '/frozen_inference_graph.pb'  #
        self.MODEL_FILE = self.MODEL_NAME + '.tar.gz'
        self.NUM_CLASSES = 90
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        #self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    def loadLabelMap(self):
        self.label_map = label_map_util.load_labelmap("../" + self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def configureModel(self):
        self.loadModel()
        self.loadLabelMap()

    # Helper code
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def model_setup(self):
        if self.model.get_bool_custom_trained():
            self.setCustomModelSettings()
        else:
            self.updateName()
            #self.downloadModel()
        self.configureModel()

    def detectOcjectsFromCamera(self):
        self.model_setup()

        cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

        # Detection
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                while True:

                    # Read frame from camera
                    ret, image_np = cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Extract image tensor
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detections
                    num_detections = self.detection_graph.get_tensor_by_name(
                        'num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    # Display output
                    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                #image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def detectOcjectsFromImagesSetup(self):
        #just some setup, detection is run with method run_inference_for_single_image
        self.model_setup()
        self.IMAGE_SIZE = (12, 8)


