import sys
import unittest
import pybboxes.functional as pbf
from sahi.utils.cv import read_image
from sahi.utils.huggingface import HuggingfaceTestConstants
MODEL_DEVICE = 'cpu'
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320
if sys.version_info >= (3, 7):

    class TestHuggingfaceDetectionModel(unittest.TestCase):

        def test_load_model(self):
            if False:
                while True:
                    i = 10
            from sahi.models.huggingface import HuggingfaceDetectionModel
            huggingface_detection_model = HuggingfaceDetectionModel(model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=True)
            self.assertNotEqual(huggingface_detection_model.model, None)

        def test_set_model(self):
            if False:
                while True:
                    i = 10
            from transformers import AutoModelForObjectDetection, AutoProcessor
            from sahi.models.huggingface import HuggingfaceDetectionModel
            huggingface_model = AutoModelForObjectDetection.from_pretrained(HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH)
            huggingface_processor = AutoProcessor.from_pretrained(HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH)
            huggingface_detection_model = HuggingfaceDetectionModel(model=huggingface_model, processor=huggingface_processor, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=True)
            self.assertNotEqual(huggingface_detection_model.model, None)

        def test_perform_inference(self):
            if False:
                print('Hello World!')
            from sahi.models.huggingface import HuggingfaceDetectionModel
            huggingface_detection_model = HuggingfaceDetectionModel(model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=True, image_size=IMAGE_SIZE)
            image_path = 'tests/data/small-vehicles1.jpeg'
            image = read_image(image_path)
            huggingface_detection_model.perform_inference(image)
            original_predictions = huggingface_detection_model.original_predictions
            (scores, cat_ids, boxes) = huggingface_detection_model.get_valid_predictions(logits=original_predictions.logits[0], pred_boxes=original_predictions.pred_boxes[0])
            for (i, box) in enumerate(boxes):
                if huggingface_detection_model.category_mapping[cat_ids[i].item()] == 'car':
                    break
            (image_height, image_width, _) = huggingface_detection_model.image_shapes[0]
            box = list(pbf.convert_bbox(box.tolist(), from_type='yolo', to_type='voc', image_size=(image_width, image_height), return_values=True))
            desired_bbox = [639, 198, 663, 218]
            predicted_bbox = list(map(int, box[:4]))
            margin = 2
            for (ind, point) in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            for score in scores:
                self.assertGreaterEqual(score.item(), CONFIDENCE_THRESHOLD)

        def test_convert_original_predictions(self):
            if False:
                i = 10
                return i + 15
            from sahi.models.huggingface import HuggingfaceDetectionModel
            huggingface_detection_model = HuggingfaceDetectionModel(model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=True, image_size=IMAGE_SIZE)
            image_path = 'tests/data/small-vehicles1.jpeg'
            image = read_image(image_path)
            huggingface_detection_model.perform_inference(image)
            huggingface_detection_model.convert_original_predictions()
            object_prediction_list = huggingface_detection_model.object_prediction_list
            self.assertEqual(len(object_prediction_list), 28)
            self.assertEqual(object_prediction_list[0].category.id, 3)
            self.assertEqual(object_prediction_list[0].category.name, 'car')
            desired_bbox = [639, 198, 24, 20]
            predicted_bbox = object_prediction_list[0].bbox.to_xywh()
            margin = 2
            for (ind, point) in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            self.assertEqual(object_prediction_list[2].category.id, 3)
            self.assertEqual(object_prediction_list[2].category.name, 'car')
            desired_bbox = [745, 169, 15, 14]
            predicted_bbox = object_prediction_list[2].bbox.to_xywh()
            for (ind, point) in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            for object_prediction in object_prediction_list:
                self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

        def test_get_prediction_huggingface(self):
            if False:
                print('Hello World!')
            from sahi.models.huggingface import HuggingfaceDetectionModel
            from sahi.predict import get_prediction
            from sahi.utils.huggingface import HuggingfaceTestConstants
            huggingface_detection_model = HuggingfaceDetectionModel(model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=False, image_size=IMAGE_SIZE)
            huggingface_detection_model.load_model()
            image_path = 'tests/data/small-vehicles1.jpeg'
            image = read_image(image_path)
            prediction_result = get_prediction(image=image, detection_model=huggingface_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None)
            object_prediction_list = prediction_result.object_prediction_list
            self.assertEqual(len(object_prediction_list), 28)
            num_person = num_truck = num_car = 0
            for object_prediction in object_prediction_list:
                if object_prediction.category.name == 'person':
                    num_person += 1
                elif object_prediction.category.name == 'truck':
                    num_truck += 1
                elif object_prediction.category.name == 'car':
                    num_car += 1
            self.assertEqual(num_person, 0)
            self.assertEqual(num_truck, 1)
            self.assertEqual(num_car, 27)

        def test_get_prediction_automodel_huggingface(self):
            if False:
                return 10
            from sahi.auto_model import AutoDetectionModel
            from sahi.predict import get_prediction
            from sahi.utils.huggingface import HuggingfaceTestConstants
            huggingface_detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface', model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=False, image_size=IMAGE_SIZE)
            huggingface_detection_model.load_model()
            image_path = 'tests/data/small-vehicles1.jpeg'
            image = read_image(image_path)
            prediction_result = get_prediction(image=image, detection_model=huggingface_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None)
            object_prediction_list = prediction_result.object_prediction_list
            self.assertEqual(len(object_prediction_list), 28)
            num_person = num_truck = num_car = 0
            for object_prediction in object_prediction_list:
                if object_prediction.category.name == 'person':
                    num_person += 1
                elif object_prediction.category.name == 'truck':
                    num_truck += 1
                elif object_prediction.category.name == 'car':
                    num_car += 1
            self.assertEqual(num_person, 0)
            self.assertEqual(num_truck, 1)
            self.assertEqual(num_car, 27)

        def test_get_sliced_prediction_huggingface(self):
            if False:
                return 10
            from sahi.models.huggingface import HuggingfaceDetectionModel
            from sahi.predict import get_sliced_prediction
            from sahi.utils.huggingface import HuggingfaceTestConstants
            huggingface_detection_model = HuggingfaceDetectionModel(model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=MODEL_DEVICE, category_remapping=None, load_at_init=False, image_size=IMAGE_SIZE)
            huggingface_detection_model.load_model()
            image_path = 'tests/data/small-vehicles1.jpeg'
            slice_height = 512
            slice_width = 512
            overlap_height_ratio = 0.1
            overlap_width_ratio = 0.2
            postprocess_type = 'GREEDYNMM'
            match_metric = 'IOS'
            match_threshold = 0.5
            class_agnostic = True
            prediction_result = get_sliced_prediction(image=image_path, detection_model=huggingface_detection_model, slice_height=slice_height, slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio, perform_standard_pred=False, postprocess_type=postprocess_type, postprocess_match_threshold=match_threshold, postprocess_match_metric=match_metric, postprocess_class_agnostic=class_agnostic)
            object_prediction_list = prediction_result.object_prediction_list
            self.assertEqual(len(object_prediction_list), 54)
            num_person = num_truck = num_car = 0
            for object_prediction in object_prediction_list:
                if object_prediction.category.name == 'person':
                    num_person += 1
                elif object_prediction.category.name == 'truck':
                    num_truck += 1
                elif object_prediction.category.name == 'car':
                    num_car += 1
            self.assertEqual(num_person, 0)
            self.assertEqual(num_truck, 5)
            self.assertEqual(num_car, 49)
if __name__ == '__main__':
    unittest.main()