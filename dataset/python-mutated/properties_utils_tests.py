from hamcrest import assert_that, calling, close_to, contains_exactly, equal_to, is_, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Suite
from deepchecks.vision.checks import ImagePropertyOutliers
from deepchecks.vision.utils.image_properties import calc_default_image_properties, default_image_properties, texture_level
from deepchecks.vision.utils.label_prediction_properties import DEFAULT_CLASSIFICATION_LABEL_PROPERTIES, DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES, DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES, DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES, DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES, DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES
from deepchecks.vision.utils.vision_properties import calc_vision_properties, validate_properties

def test_calc_properties(coco_visiondata_train):
    if False:
        print('Hello World!')
    images = next(iter(coco_visiondata_train.batch_loader))['images']
    results = calc_vision_properties(images, default_image_properties)
    assert_that(results.keys(), contains_exactly('Aspect Ratio', 'Area', 'Brightness', 'RMS Contrast', 'Mean Red Relative Intensity', 'Mean Green Relative Intensity', 'Mean Blue Relative Intensity'))
    assert_that(sum(results['Brightness']), close_to(15.56, 0.01))

def test_calc_default_image_properties(coco_visiondata_train):
    if False:
        for i in range(10):
            print('nop')
    images = next(iter(coco_visiondata_train.batch_loader))['images']
    results = calc_default_image_properties(images)
    assert_that(results.keys(), contains_exactly('Aspect Ratio', 'Area', 'Brightness', 'RMS Contrast', 'Mean Red Relative Intensity', 'Mean Green Relative Intensity', 'Mean Blue Relative Intensity'))
    assert_that(sum(results['Brightness']), close_to(15.563, 0.01))

def test_calc_default_image_properties_grayscale(mnist_visiondata_train):
    if False:
        for i in range(10):
            print('nop')
    images = next(iter(mnist_visiondata_train.batch_loader))['images']
    results = calc_default_image_properties(images)
    assert_that(results['Mean Red Relative Intensity'][0], is_(None))
    assert_that(sum(results['Brightness']), close_to(2069.19, 0.01))

def test_default_properties():
    if False:
        i = 10
        return i + 15
    validate_properties(default_image_properties)
    validate_properties(DEFAULT_CLASSIFICATION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES)
    validate_properties(DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES)
    validate_properties(DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES)
    validate_properties(DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES)

def test_validate_properties_with_instance_of_incorrect_type_provided():
    if False:
        for i in range(10):
            print('nop')
    assert_that(calling(validate_properties).with_args(object()), detects_incorrect_type_of_input())

def test_validate_properties_with_empty_properties_list():
    if False:
        for i in range(10):
            print('nop')
    assert_that(calling(validate_properties).with_args([]), detects_empty_list())

def test_validate_properties_with_unsupported_item_type():
    if False:
        return 10
    properties = [*default_image_properties, object()]
    msg = "List of properties contains next problems:\\n\\+ Item #7: property must be of type dict, and include keys \\('name', 'method', 'output_type'\\)."
    assert_that(calling(validate_properties).with_args(properties), raises(DeepchecksValueError, msg))

def test_validate_properties_with_incorrect_property_dict_structure():
    if False:
        for i in range(10):
            print('nop')
    property = default_image_properties[0].copy()
    property.pop('method')
    msg = "List of properties contains next problems:\\n\\+ Item #0: property must be of type dict, and include keys \\('name', 'method', 'output_type'\\)."
    assert_that(calling(validate_properties).with_args([property]), raises(DeepchecksValueError, msg))

def test_validate_properties_with_bad_name_field():
    if False:
        while True:
            i = 10

    def prop(predictions):
        if False:
            print('Hello World!')
        return [int(x[0][0]) if len(x) != 0 else 0 for x in predictions]
    assert_that(calling(validate_properties).with_args([{'name': 'test', 'method': prop, 'output_type': 'continuous'}]), raises(DeepchecksValueError, 'List of properties contains next problems:\\n\\+ Property test: field \\"output_type\\" must be one of \\(\'categorical\', \'numerical\', \'class_id\'\\), instead got continuous'))
    assert_that(calling(validate_properties).with_args([{'name234': 'test', 'method': prop, 'output_type': 'numerical'}]), raises(DeepchecksValueError, "List of properties contains next problems:\\n\\+ Item #0: property must be of type dict, and include keys \\('name', 'method', 'output_type'\\)."))

def test_validate_properties_with_incorrect_property_output_type():
    if False:
        return 10
    property = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES[0].copy()
    property['output_type'] = 'hello-world'
    assert_that(calling(validate_properties).with_args([property]), raises(DeepchecksValueError, f"""List of properties contains next problems:\\n\\+ Property {property['name']}: field \\"output_type\\" must be one of \\('categorical', 'numerical', 'class_id'\\), instead got hello-world"""))

def detects_empty_list():
    if False:
        for i in range(10):
            print('nop')
    return raises(DeepchecksValueError, "Properties list can't be empty")

def detects_incorrect_type_of_input(provided_value_type: str='object'):
    if False:
        while True:
            i = 10
    return raises(DeepchecksValueError, f'Expected properties to be a list, instead got {provided_value_type}')

def test_sharpness_and_texture_level(coco_visiondata_train):
    if False:
        print('Hello World!')
    props = [{'name': 'texture', 'method': texture_level, 'output_type': 'continuous'}]
    images = next(iter(coco_visiondata_train.batch_loader))['images']
    results = calc_vision_properties(images, props)
    assert_that(sum(results['texture']), close_to(1.79, 0.01))

def test_suite_different_properties_per_check(coco_visiondata_train):
    if False:
        for i in range(10):
            print('nop')
    props = [{'name': 'texture', 'method': texture_level, 'output_type': 'numerical'}]
    check1 = ImagePropertyOutliers(image_properties=props)
    check2 = ImagePropertyOutliers()
    suite = Suite('prop_suite', check1, check2)
    result = suite.run(coco_visiondata_train)
    assert_that(list(result.results[0].value.keys()), contains_exactly('texture'))
    assert_that(sorted(result.results[1].value.keys()), equal_to(sorted((x['name'] for x in default_image_properties))))