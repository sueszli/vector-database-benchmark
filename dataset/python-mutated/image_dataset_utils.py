import numpy as np
from keras.api_export import keras_export
from keras.backend.config import standardize_data_format
from keras.utils import dataset_utils
from keras.utils import image_utils
from keras.utils.module_utils import tensorflow as tf
ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

@keras_export(['keras.utils.image_dataset_from_directory', 'keras.preprocessing.image_dataset_from_directory'])
def image_dataset_from_directory(directory, labels='inferred', label_mode='int', class_names=None, color_mode='rgb', batch_size=32, image_size=(256, 256), shuffle=True, seed=None, validation_split=None, subset=None, interpolation='bilinear', follow_links=False, crop_to_aspect_ratio=False, data_format=None):
    if False:
        return 10
    'Generates a `tf.data.Dataset` from image files in a directory.\n\n    If your directory structure is:\n\n    ```\n    main_directory/\n    ...class_a/\n    ......a_image_1.jpg\n    ......a_image_2.jpg\n    ...class_b/\n    ......b_image_1.jpg\n    ......b_image_2.jpg\n    ```\n\n    Then calling `image_dataset_from_directory(main_directory,\n    labels=\'inferred\')` will return a `tf.data.Dataset` that yields batches of\n    images from the subdirectories `class_a` and `class_b`, together with labels\n    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).\n\n    Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.\n    Animated gifs are truncated to the first frame.\n\n    Args:\n        directory: Directory where the data is located.\n            If `labels` is `"inferred"`, it should contain\n            subdirectories, each containing images for a class.\n            Otherwise, the directory structure is ignored.\n        labels: Either `"inferred"`\n            (labels are generated from the directory structure),\n            `None` (no labels),\n            or a list/tuple of integer labels of the same size as the number of\n            image files found in the directory. Labels should be sorted\n            according to the alphanumeric order of the image file paths\n            (obtained via `os.walk(directory)` in Python).\n        label_mode: String describing the encoding of `labels`. Options are:\n            - `"int"`: means that the labels are encoded as integers\n                (e.g. for `sparse_categorical_crossentropy` loss).\n            - `"categorical"` means that the labels are\n                encoded as a categorical vector\n                (e.g. for `categorical_crossentropy` loss).\n            - `"binary"` means that the labels (there can be only 2)\n                are encoded as `float32` scalars with values 0 or 1\n                (e.g. for `binary_crossentropy`).\n            - `None` (no labels).\n        class_names: Only valid if `labels` is `"inferred"`.\n            This is the explicit list of class names\n            (must match names of subdirectories). Used to control the order\n            of the classes (otherwise alphanumerical order is used).\n        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`.\n            Defaults to `"rgb"`. Whether the images will be converted to\n            have 1, 3, or 4 channels.\n        batch_size: Size of the batches of data. Defaults to 32.\n            If `None`, the data will not be batched\n            (the dataset will yield individual samples).\n        image_size: Size to resize images to after they are read from disk,\n            specified as `(height, width)`. Defaults to `(256, 256)`.\n            Since the pipeline processes batches of images that must all have\n            the same size, this must be provided.\n        shuffle: Whether to shuffle the data. Defaults to `True`.\n            If set to `False`, sorts the data in alphanumeric order.\n        seed: Optional random seed for shuffling and transformations.\n        validation_split: Optional float between 0 and 1,\n            fraction of data to reserve for validation.\n        subset: Subset of the data to return.\n            One of `"training"`, `"validation"`, or `"both"`.\n            Only used if `validation_split` is set.\n            When `subset="both"`, the utility returns a tuple of two datasets\n            (the training and validation datasets respectively).\n        interpolation: String, the interpolation method used when\n            resizing images. Defaults to `"bilinear"`.\n            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,\n            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.\n        follow_links: Whether to visit subdirectories pointed to by symlinks.\n            Defaults to `False`.\n        crop_to_aspect_ratio: If `True`, resize the images without aspect\n            ratio distortion. When the original aspect ratio differs from the\n            target aspect ratio, the output image will be cropped so as to\n            return the largest possible window in the image\n            (of size `image_size`) that matches the target aspect ratio. By\n            default (`crop_to_aspect_ratio=False`), aspect ratio may not be\n            preserved.\n        data_format: If None uses keras.config.image_data_format()\n            otherwise either \'channel_last\' or \'channel_first\'.\n\n    Returns:\n\n    A `tf.data.Dataset` object.\n\n    - If `label_mode` is `None`, it yields `float32` tensors of shape\n        `(batch_size, image_size[0], image_size[1], num_channels)`,\n        encoding images (see below for rules regarding `num_channels`).\n    - Otherwise, it yields a tuple `(images, labels)`, where `images` has\n        shape `(batch_size, image_size[0], image_size[1], num_channels)`,\n        and `labels` follows the format described below.\n\n    Rules regarding labels format:\n\n    - if `label_mode` is `"int"`, the labels are an `int32` tensor of shape\n        `(batch_size,)`.\n    - if `label_mode` is `"binary"`, the labels are a `float32` tensor of\n        1s and 0s of shape `(batch_size, 1)`.\n    - if `label_mode` is `"categorical"`, the labels are a `float32` tensor\n        of shape `(batch_size, num_classes)`, representing a one-hot\n        encoding of the class index.\n\n    Rules regarding number of channels in the yielded images:\n\n    - if `color_mode` is `"grayscale"`,\n        there\'s 1 channel in the image tensors.\n    - if `color_mode` is `"rgb"`,\n        there are 3 channels in the image tensors.\n    - if `color_mode` is `"rgba"`,\n        there are 4 channels in the image tensors.\n    '
    if labels not in ('inferred', None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(f'`labels` argument should be a list/tuple of integer labels, of the same size as the number of image files in the target directory. If you wish to infer the labels from the subdirectory names in the target directory, pass `labels="inferred"`. If you wish to get a dataset that only contains images (no labels), pass `labels=None`. Received: labels={labels}')
        if class_names:
            raise ValueError(f'You can only pass `class_names` if `labels="inferred"`. Received: labels={labels}, and class_names={class_names}')
    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(f'`label_mode` argument must be one of "int", "categorical", "binary", or None. Received: label_mode={label_mode}')
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(f'`color_mode` must be one of {{"rgb", "rgba", "grayscale"}}. Received: color_mode={color_mode}')
    interpolation = interpolation.lower()
    supported_interpolations = ('bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic')
    if interpolation not in supported_interpolations:
        raise ValueError(f'Argument `interpolation` should be one of {supported_interpolations}. Received: interpolation={interpolation}')
    dataset_utils.check_validation_split_arg(validation_split, subset, shuffle, seed)
    if seed is None:
        seed = np.random.randint(1000000.0)
    (image_paths, labels, class_names) = dataset_utils.index_directory(directory, labels, formats=ALLOWLIST_FORMATS, class_names=class_names, shuffle=shuffle, seed=seed, follow_links=follow_links)
    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(f'When passing `label_mode="binary"`, there must be exactly 2 class_names. Received: class_names={class_names}')
    data_format = standardize_data_format(data_format=data_format)
    if subset == 'both':
        (image_paths_train, labels_train) = dataset_utils.get_training_or_validation_split(image_paths, labels, validation_split, 'training')
        (image_paths_val, labels_val) = dataset_utils.get_training_or_validation_split(image_paths, labels, validation_split, 'validation')
        if not image_paths_train:
            raise ValueError(f'No training images found in directory {directory}. Allowed formats: {ALLOWLIST_FORMATS}')
        if not image_paths_val:
            raise ValueError(f'No validation images found in directory {directory}. Allowed formats: {ALLOWLIST_FORMATS}')
        train_dataset = paths_and_labels_to_dataset(image_paths=image_paths_train, image_size=image_size, num_channels=num_channels, labels=labels_train, label_mode=label_mode, num_classes=len(class_names) if class_names else 0, interpolation=interpolation, crop_to_aspect_ratio=crop_to_aspect_ratio, data_format=data_format)
        val_dataset = paths_and_labels_to_dataset(image_paths=image_paths_val, image_size=image_size, num_channels=num_channels, labels=labels_val, label_mode=label_mode, num_classes=len(class_names) if class_names else 0, interpolation=interpolation, crop_to_aspect_ratio=crop_to_aspect_ratio, data_format=data_format)
        if batch_size is not None:
            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
            train_dataset = train_dataset.batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)
        elif shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=1024, seed=seed)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset.class_names = class_names
        val_dataset.class_names = class_names
        train_dataset.file_paths = image_paths_train
        val_dataset.file_paths = image_paths_val
        dataset = [train_dataset, val_dataset]
    else:
        (image_paths, labels) = dataset_utils.get_training_or_validation_split(image_paths, labels, validation_split, subset)
        if not image_paths:
            raise ValueError(f'No images found in directory {directory}. Allowed formats: {ALLOWLIST_FORMATS}')
        dataset = paths_and_labels_to_dataset(image_paths=image_paths, image_size=image_size, num_channels=num_channels, labels=labels, label_mode=label_mode, num_classes=len(class_names) if class_names else 0, interpolation=interpolation, crop_to_aspect_ratio=crop_to_aspect_ratio, data_format=data_format)
        if batch_size is not None:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
            dataset = dataset.batch(batch_size)
        elif shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset.class_names = class_names
        dataset.file_paths = image_paths
    return dataset

def paths_and_labels_to_dataset(image_paths, image_size, num_channels, labels, label_mode, num_classes, interpolation, data_format, crop_to_aspect_ratio=False):
    if False:
        i = 10
        return i + 15
    'Constructs a dataset of images and labels.'
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    args = (image_size, num_channels, interpolation, data_format, crop_to_aspect_ratio)
    img_ds = path_ds.map(lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        img_ds = tf.data.Dataset.zip((img_ds, label_ds))
    return img_ds

def load_image(path, image_size, num_channels, interpolation, data_format, crop_to_aspect_ratio=False):
    if False:
        return 10
    'Load an image from a path and resize it.'
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
    if crop_to_aspect_ratio:
        from keras.backend import tensorflow as tf_backend
        if data_format == 'channels_first':
            img = tf.transpose(img, (2, 0, 1))
        img = image_utils.smart_resize(img, image_size, interpolation=interpolation, data_format=data_format, backend_module=tf_backend)
    else:
        img = tf.image.resize(img, image_size, method=interpolation)
        if data_format == 'channels_first':
            img = tf.transpose(img, (2, 0, 1))
    if data_format == 'channels_last':
        img.set_shape((image_size[0], image_size[1], num_channels))
    else:
        img.set_shape((num_channels, image_size[0], image_size[1]))
    return img