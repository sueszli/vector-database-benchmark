import random as _random
import turicreate as _tc
from turicreate import extensions as _extensions
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits.one_shot_object_detector.util._error_handling import check_one_shot_input
from turicreate.toolkits import _data_zoo

def preview_synthetic_training_data(data, target, backgrounds=None, verbose=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    A utility function to visualize the synthetically generated data.\n\n    Parameters\n    ----------\n    data : SFrame | tc.Image\n        A single starter image or an SFrame that contains the starter images\n        along with their corresponding labels.  These image(s) can be in either\n        RGB or RGBA format. They should not be padded.\n\n    target : string\n        Name of the target (when data is a single image) or the target column\n        name (when data is an SFrame of images).\n\n    backgrounds : optional SArray\n        A list of backgrounds used for synthetic data generation. When set to\n        None, a set of default backgrounds are downloaded and used.\n\n    verbose : bool optional\n        If True, print progress updates and details.\n\n    Returns\n    -------\n    out : SFrame\n        An SFrame of sythetically generated annotated training data.\n    '
    (dataset_to_augment, image_column_name, target_column_name) = check_one_shot_input(data, target, backgrounds)
    _tkutl._handle_missing_values(dataset_to_augment, image_column_name, 'dataset')
    if backgrounds is None:
        backgrounds_downloader = _data_zoo.OneShotObjectDetectorBackgroundData()
        backgrounds = backgrounds_downloader.get_backgrounds()
        backgrounds = backgrounds.apply(lambda im: _tc.image_analysis.resize(im, int(im.width / 2), int(im.height / 2), im.channels))
    seed = kwargs['seed'] if 'seed' in kwargs else _random.randint(0, 2 ** 32 - 1)
    options_for_augmentation = {'seed': seed, 'verbose': verbose}
    one_shot_model = _extensions.one_shot_object_detector()
    augmented_data = one_shot_model.augment(dataset_to_augment, image_column_name, target_column_name, backgrounds, options_for_augmentation)
    return augmented_data