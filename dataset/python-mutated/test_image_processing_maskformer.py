import unittest
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_torch_available():
    import torch
    if is_vision_available():
        from transformers import MaskFormerImageProcessor
        from transformers.models.maskformer.image_processing_maskformer import binary_mask_to_rle
        from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput
if is_vision_available():
    from PIL import Image

class MaskFormerImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, min_resolution=30, max_resolution=400, size=None, do_resize=True, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], num_labels=10, do_reduce_labels=True, ignore_index=255):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = {'shortest_edge': 32, 'longest_edge': 1333} if size is None else size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisor = 0
        self.batch_size = 2
        self.num_queries = 3
        self.num_classes = 2
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.do_reduce_labels = do_reduce_labels
        self.ignore_index = ignore_index

    def prepare_image_processor_dict(self):
        if False:
            print('Hello World!')
        return {'do_resize': self.do_resize, 'size': self.size, 'do_normalize': self.do_normalize, 'image_mean': self.image_mean, 'image_std': self.image_std, 'size_divisor': self.size_divisor, 'num_labels': self.num_labels, 'do_reduce_labels': self.do_reduce_labels, 'ignore_index': self.ignore_index}

    def get_expected_values(self, image_inputs, batched=False):
        if False:
            i = 10
            return i + 15
        '\n        This function computes the expected height and width when providing images to MaskFormerImageProcessor,\n        assuming do_resize is set to True with a scalar size.\n        '
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                (w, h) = image.size
            else:
                (h, w) = (image.shape[1], image.shape[2])
            if w < h:
                expected_height = int(self.size['shortest_edge'] * h / w)
                expected_width = self.size['shortest_edge']
            elif w > h:
                expected_height = self.size['shortest_edge']
                expected_width = int(self.size['shortest_edge'] * w / h)
            else:
                expected_height = self.size['shortest_edge']
                expected_width = self.size['shortest_edge']
        else:
            expected_values = []
            for image in image_inputs:
                (expected_height, expected_width) = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]
        return (expected_height, expected_width)

    def get_fake_maskformer_outputs(self):
        if False:
            while True:
                i = 10
        return MaskFormerForInstanceSegmentationOutput(class_queries_logits=torch.randn((self.batch_size, self.num_queries, self.num_classes + 1)), masks_queries_logits=torch.randn((self.batch_size, self.num_queries, self.height, self.width)))

    def expected_output_image_shape(self, images):
        if False:
            print('Hello World!')
        (height, width) = self.get_expected_values(images, batched=True)
        return (self.num_channels, height, width)

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            i = 10
            return i + 15
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class MaskFormerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = MaskFormerImageProcessor if is_vision_available() and is_torch_available() else None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.image_processor_tester = MaskFormerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            while True:
                i = 10
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            return 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'do_resize'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'ignore_index'))
        self.assertTrue(hasattr(image_processing, 'num_labels'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            return 10
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'shortest_edge': 32, 'longest_edge': 1333})
        self.assertEqual(image_processor.size_divisor, 0)
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, max_size=84, size_divisibility=8)
        self.assertEqual(image_processor.size, {'shortest_edge': 42, 'longest_edge': 84})
        self.assertEqual(image_processor.size_divisor, 8)

    def comm_get_image_processing_inputs(self, with_segmentation_maps=False, is_instance_map=False, segmentation_type='np'):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        num_labels = self.image_processor_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = dict(enumerate(labels_expanded))
            annotations = [np.random.randint(0, high * 2, (img.size[1], img.size[0])).astype(np.uint8) for img in image_inputs]
            if segmentation_type == 'pil':
                annotations = [Image.fromarray(annotation) for annotation in annotations]
        inputs = image_processing(image_inputs, annotations, return_tensors='pt', instance_id_to_semantic_id=instance_id_to_semantic_id, pad_and_return_pixel_mask=True)
        return inputs

    def test_with_size_divisor(self):
        if False:
            return 10
        size_divisors = [8, 16, 32]
        weird_input_sizes = [(407, 802), (582, 1094)]
        for size_divisor in size_divisors:
            image_processor_dict = {**self.image_processor_dict, **{'size_divisor': size_divisor}}
            image_processing = self.image_processing_class(**image_processor_dict)
            for weird_input_size in weird_input_sizes:
                inputs = image_processing([np.ones((3, *weird_input_size))], return_tensors='pt')
                pixel_values = inputs['pixel_values']
                self.assertTrue(pixel_values.shape[-1] % size_divisor == 0)
                self.assertTrue(pixel_values.shape[-2] % size_divisor == 0)

    def test_call_with_segmentation_maps(self):
        if False:
            print('Hello World!')

        def common(is_instance_map=False, segmentation_type=None):
            if False:
                for i in range(10):
                    print('nop')
            inputs = self.comm_get_image_processing_inputs(with_segmentation_maps=True, is_instance_map=is_instance_map, segmentation_type=segmentation_type)
            mask_labels = inputs['mask_labels']
            class_labels = inputs['class_labels']
            pixel_values = inputs['pixel_values']
            for (mask_label, class_label) in zip(mask_labels, class_labels):
                self.assertEqual(mask_label.shape[0], class_label.shape[0])
                self.assertEqual(mask_label.shape[1:], pixel_values.shape[2:])
        common()
        common(is_instance_map=True)
        common(is_instance_map=False, segmentation_type='pil')
        common(is_instance_map=True, segmentation_type='pil')

    def test_integration_instance_segmentation(self):
        if False:
            for i in range(10):
                print('nop')
        repo_id = 'nielsr/image-segmentation-toy-data'
        image1 = Image.open(hf_hub_download(repo_id=repo_id, filename='instance_segmentation_image_1.png', repo_type='dataset'))
        image2 = Image.open(hf_hub_download(repo_id=repo_id, filename='instance_segmentation_image_2.png', repo_type='dataset'))
        annotation1 = Image.open(hf_hub_download(repo_id=repo_id, filename='instance_segmentation_annotation_1.png', repo_type='dataset'))
        annotation2 = Image.open(hf_hub_download(repo_id=repo_id, filename='instance_segmentation_annotation_2.png', repo_type='dataset'))

        def get_instance_segmentation_and_mapping(annotation):
            if False:
                i = 10
                return i + 15
            instance_seg = np.array(annotation)[:, :, 1]
            class_id_map = np.array(annotation)[:, :, 0]
            class_labels = np.unique(class_id_map)
            inst2class = {}
            for label in class_labels:
                instance_ids = np.unique(instance_seg[class_id_map == label])
                inst2class.update({i: label for i in instance_ids})
            return (instance_seg, inst2class)
        (instance_seg1, inst2class1) = get_instance_segmentation_and_mapping(annotation1)
        (instance_seg2, inst2class2) = get_instance_segmentation_and_mapping(annotation2)
        image_processing = MaskFormerImageProcessor(reduce_labels=True, ignore_index=255, size=(512, 512))
        inputs = image_processing([image1, image2], [instance_seg1, instance_seg2], instance_id_to_semantic_id=[inst2class1, inst2class2], return_tensors='pt')
        self.assertEqual(inputs['pixel_values'].shape, (2, 3, 512, 512))
        self.assertEqual(inputs['pixel_mask'].shape, (2, 512, 512))
        self.assertEqual(len(inputs['class_labels']), 2)
        self.assertTrue(torch.allclose(inputs['class_labels'][0], torch.tensor([30, 55])))
        self.assertTrue(torch.allclose(inputs['class_labels'][1], torch.tensor([4, 4, 23, 55])))
        self.assertEqual(len(inputs['mask_labels']), 2)
        self.assertEqual(inputs['mask_labels'][0].shape, (2, 512, 512))
        self.assertEqual(inputs['mask_labels'][1].shape, (4, 512, 512))
        self.assertEquals(inputs['mask_labels'][0].sum().item(), 41527.0)
        self.assertEquals(inputs['mask_labels'][1].sum().item(), 26259.0)

    def test_integration_semantic_segmentation(self):
        if False:
            i = 10
            return i + 15
        repo_id = 'nielsr/image-segmentation-toy-data'
        image1 = Image.open(hf_hub_download(repo_id=repo_id, filename='semantic_segmentation_image_1.png', repo_type='dataset'))
        image2 = Image.open(hf_hub_download(repo_id=repo_id, filename='semantic_segmentation_image_2.png', repo_type='dataset'))
        annotation1 = Image.open(hf_hub_download(repo_id=repo_id, filename='semantic_segmentation_annotation_1.png', repo_type='dataset'))
        annotation2 = Image.open(hf_hub_download(repo_id=repo_id, filename='semantic_segmentation_annotation_2.png', repo_type='dataset'))
        image_processing = MaskFormerImageProcessor(reduce_labels=True, ignore_index=255, size=(512, 512))
        inputs = image_processing([image1, image2], [annotation1, annotation2], return_tensors='pt')
        self.assertEqual(inputs['pixel_values'].shape, (2, 3, 512, 512))
        self.assertEqual(inputs['pixel_mask'].shape, (2, 512, 512))
        self.assertEqual(len(inputs['class_labels']), 2)
        self.assertTrue(torch.allclose(inputs['class_labels'][0], torch.tensor([2, 4, 60])))
        self.assertTrue(torch.allclose(inputs['class_labels'][1], torch.tensor([0, 3, 7, 8, 15, 28, 30, 143])))
        self.assertEqual(len(inputs['mask_labels']), 2)
        self.assertEqual(inputs['mask_labels'][0].shape, (3, 512, 512))
        self.assertEqual(inputs['mask_labels'][1].shape, (8, 512, 512))
        self.assertEquals(inputs['mask_labels'][0].sum().item(), 170200.0)
        self.assertEquals(inputs['mask_labels'][1].sum().item(), 257036.0)

    def test_integration_panoptic_segmentation(self):
        if False:
            while True:
                i = 10
        dataset = load_dataset('nielsr/ade20k-panoptic-demo')
        image1 = dataset['train'][0]['image']
        image2 = dataset['train'][1]['image']
        segments_info1 = dataset['train'][0]['segments_info']
        segments_info2 = dataset['train'][1]['segments_info']
        annotation1 = dataset['train'][0]['label']
        annotation2 = dataset['train'][1]['label']

        def rgb_to_id(color):
            if False:
                return 10
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            if False:
                while True:
                    i = 10
            annotation = np.array(annotation)
            panoptic_map = rgb_to_id(annotation)
            inst2class = {segment['id']: segment['category_id'] for segment in segments_info}
            return (panoptic_map, inst2class)
        (panoptic_map1, inst2class1) = create_panoptic_map(annotation1, segments_info1)
        (panoptic_map2, inst2class2) = create_panoptic_map(annotation2, segments_info2)
        image_processing = MaskFormerImageProcessor(ignore_index=0, do_resize=False)
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = image_processing.encode_inputs(pixel_values_list, [panoptic_map1, panoptic_map2], instance_id_to_semantic_id=[inst2class1, inst2class2], return_tensors='pt')
        self.assertEqual(inputs['pixel_values'].shape, (2, 3, 512, 711))
        self.assertEqual(inputs['pixel_mask'].shape, (2, 512, 711))
        self.assertEqual(len(inputs['class_labels']), 2)
        expected_class_labels = torch.tensor([4, 17, 32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 3, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5, 12, 12, 12, 12, 12, 12, 12, 0, 43, 43, 43, 96, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])
        self.assertTrue(torch.allclose(inputs['class_labels'][0], torch.tensor(expected_class_labels)))
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 5, 12, 12, 0, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])
        self.assertTrue(torch.allclose(inputs['class_labels'][1], expected_class_labels))
        self.assertEqual(len(inputs['mask_labels']), 2)
        self.assertEqual(inputs['mask_labels'][0].shape, (79, 512, 711))
        self.assertEqual(inputs['mask_labels'][1].shape, (61, 512, 711))
        self.assertEquals(inputs['mask_labels'][0].sum().item(), 315193.0)
        self.assertEquals(inputs['mask_labels'][1].sum().item(), 350747.0)

    def test_binary_mask_to_rle(self):
        if False:
            i = 10
            return i + 15
        fake_binary_mask = np.zeros((20, 50))
        fake_binary_mask[0, 20:] = 1
        fake_binary_mask[1, :15] = 1
        fake_binary_mask[5, :10] = 1
        rle = binary_mask_to_rle(fake_binary_mask)
        self.assertEqual(len(rle), 4)
        self.assertEqual(rle[0], 21)
        self.assertEqual(rle[1], 45)

    def test_post_process_segmentation(self):
        if False:
            i = 10
            return i + 15
        fature_extractor = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_maskformer_outputs()
        segmentation = fature_extractor.post_process_segmentation(outputs)
        self.assertEqual(segmentation.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_classes, self.image_processor_tester.height, self.image_processor_tester.width))
        target_size = (1, 4)
        segmentation = fature_extractor.post_process_segmentation(outputs, target_size=target_size)
        self.assertEqual(segmentation.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_classes, *target_size))

    def test_post_process_semantic_segmentation(self):
        if False:
            print('Hello World!')
        fature_extractor = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_maskformer_outputs()
        segmentation = fature_extractor.post_process_semantic_segmentation(outputs)
        self.assertEqual(len(segmentation), self.image_processor_tester.batch_size)
        self.assertEqual(segmentation[0].shape, (self.image_processor_tester.height, self.image_processor_tester.width))
        target_sizes = [(1, 4) for i in range(self.image_processor_tester.batch_size)]
        segmentation = fature_extractor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        self.assertEqual(segmentation[0].shape, target_sizes[0])

    def test_post_process_instance_segmentation(self):
        if False:
            while True:
                i = 10
        image_processor = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_maskformer_outputs()
        segmentation = image_processor.post_process_instance_segmentation(outputs, threshold=0)
        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_size)
        for el in segmentation:
            self.assertTrue('segmentation' in el)
            self.assertTrue('segments_info' in el)
            self.assertEqual(type(el['segments_info']), list)
            self.assertEqual(el['segmentation'].shape, (self.image_processor_tester.height, self.image_processor_tester.width))
        segmentation = image_processor.post_process_instance_segmentation(outputs, threshold=0, return_binary_maps=True)
        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_size)
        for el in segmentation:
            self.assertTrue('segmentation' in el)
            self.assertTrue('segments_info' in el)
            self.assertEqual(type(el['segments_info']), list)
            self.assertEqual(len(el['segmentation'].shape), 3)
            self.assertEqual(el['segmentation'].shape[1:], (self.image_processor_tester.height, self.image_processor_tester.width))

    def test_post_process_panoptic_segmentation(self):
        if False:
            i = 10
            return i + 15
        image_processing = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_maskformer_outputs()
        segmentation = image_processing.post_process_panoptic_segmentation(outputs, threshold=0)
        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_size)
        for el in segmentation:
            self.assertTrue('segmentation' in el)
            self.assertTrue('segments_info' in el)
            self.assertEqual(type(el['segments_info']), list)
            self.assertEqual(el['segmentation'].shape, (self.image_processor_tester.height, self.image_processor_tester.width))

    def test_post_process_label_fusing(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class(num_labels=self.image_processor_tester.num_classes)
        outputs = self.image_processor_tester.get_fake_maskformer_outputs()
        segmentation = image_processor.post_process_panoptic_segmentation(outputs, threshold=0, mask_threshold=0, overlap_mask_area_threshold=0)
        unfused_segments = [el['segments_info'] for el in segmentation]
        fused_segmentation = image_processor.post_process_panoptic_segmentation(outputs, threshold=0, mask_threshold=0, overlap_mask_area_threshold=0, label_ids_to_fuse={1})
        fused_segments = [el['segments_info'] for el in fused_segmentation]
        for (el_unfused, el_fused) in zip(unfused_segments, fused_segments):
            if len(el_unfused) == 0:
                self.assertEqual(len(el_unfused), len(el_fused))
                continue
            fuse_targets = [1 for el in el_unfused if el['label_id'] in {1}]
            num_to_fuse = 0 if len(fuse_targets) == 0 else sum(fuse_targets) - 1
            expected_num_segments = max([el['id'] for el in el_unfused]) - num_to_fuse
            num_segments_fused = max([el['id'] for el in el_fused])
            self.assertEqual(num_segments_fused, expected_num_segments)