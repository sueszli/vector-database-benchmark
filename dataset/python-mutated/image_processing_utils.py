import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension
from .utils import IMAGE_PROCESSOR_NAME, PushToHubMixin, add_model_info_to_auto_map, cached_file, copy_func, download_url, is_offline_mode, is_remote_url, is_vision_available, logging
if is_vision_available():
    from PIL import Image
logger = logging.get_logger(__name__)

class BatchFeature(BaseBatchFeature):
    """
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

class ImageProcessingMixin(PushToHubMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """
    _auto_class = None

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Set elements of `kwargs` as attributes.'
        self._processor_class = kwargs.pop('processor_class', None)
        for (key, value) in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        if False:
            while True:
                i = 10
        'Sets processor class as an attribute.'
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, local_files_only: bool=False, token: Optional[Union[str, bool]]=None, revision: str='main', **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a type of [`~image_processing_utils.ImageProcessingMixin`] from an image processor.\n\n        Args:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                This can be either:\n\n                - a string, the *model id* of a pretrained image_processor hosted inside a model repo on\n                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or\n                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.\n                - a path to a *directory* containing a image processor file saved using the\n                  [`~image_processing_utils.ImageProcessingMixin.save_pretrained`] method, e.g.,\n                  `./my_model_directory/`.\n                - a path or url to a saved image processor JSON *file*, e.g.,\n                  `./my_model_directory/preprocessor_config.json`.\n            cache_dir (`str` or `os.PathLike`, *optional*):\n                Path to a directory in which a downloaded pretrained model image processor should be cached if the\n                standard cache should not be used.\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force to (re-)download the image processor files and override the cached versions if\n                they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received file. Attempts to resume the download if such a file\n                exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}.` The proxies are used on each request.\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n\n\n                <Tip>\n\n                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n                </Tip>\n\n            return_unused_kwargs (`bool`, *optional*, defaults to `False`):\n                If `False`, then this function returns just the final image processor object. If `True`, then this\n                functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary\n                consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of\n                `kwargs` which has not been used to update `image_processor` and is otherwise ignored.\n            subfolder (`str`, *optional*, defaults to `""`):\n                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n                specify the folder name here.\n            kwargs (`Dict[str, Any]`, *optional*):\n                The values in kwargs of any keys which are image processor attributes will be used to override the\n                loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is\n                controlled by the `return_unused_kwargs` keyword parameter.\n\n        Returns:\n            A image processor of type [`~image_processing_utils.ImageProcessingMixin`].\n\n        Examples:\n\n        ```python\n        # We can\'t instantiate directly the base class *ImageProcessingMixin* so let\'s show the examples on a\n        # derived class: *CLIPImageProcessor*\n        image_processor = CLIPImageProcessor.from_pretrained(\n            "openai/clip-vit-base-patch32"\n        )  # Download image_processing_config from huggingface.co and cache.\n        image_processor = CLIPImageProcessor.from_pretrained(\n            "./test/saved_model/"\n        )  # E.g. image processor (or model) was saved using *save_pretrained(\'./test/saved_model/\')*\n        image_processor = CLIPImageProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")\n        image_processor = CLIPImageProcessor.from_pretrained(\n            "openai/clip-vit-base-patch32", do_normalize=False, foo=False\n        )\n        assert image_processor.do_normalize is False\n        image_processor, unused_kwargs = CLIPImageProcessor.from_pretrained(\n            "openai/clip-vit-base-patch32", do_normalize=False, foo=False, return_unused_kwargs=True\n        )\n        assert image_processor.do_normalize is False\n        assert unused_kwargs == {"foo": False}\n        ```'
        kwargs['cache_dir'] = cache_dir
        kwargs['force_download'] = force_download
        kwargs['local_files_only'] = local_files_only
        kwargs['revision'] = revision
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            kwargs['token'] = token
        (image_processor_dict, kwargs) = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(image_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the\n        [`~image_processing_utils.ImageProcessingMixin.from_pretrained`] class method.\n\n        Args:\n            save_directory (`str` or `os.PathLike`):\n                Directory where the image processor JSON file will be saved (will be created if it does not exist).\n            push_to_hub (`bool`, *optional*, defaults to `False`):\n                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the\n                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your\n                namespace).\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.\n        '
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if kwargs.get('token', None) is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            kwargs['token'] = use_auth_token
        if os.path.isfile(save_directory):
            raise AssertionError(f'Provided path ({save_directory}) should be a directory, not a file')
        os.makedirs(save_directory, exist_ok=True)
        if push_to_hub:
            commit_message = kwargs.pop('commit_message', None)
            repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)
        self.to_json_file(output_image_processor_file)
        logger.info(f'Image processor saved in {output_image_processor_file}')
        if push_to_hub:
            self._upload_modified_files(save_directory, repo_id, files_timestamps, commit_message=commit_message, token=kwargs.get('token'))
        return [output_image_processor_file]

    @classmethod
    def get_image_processor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a\n        image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.\n\n        Parameters:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.\n            subfolder (`str`, *optional*, defaults to `""`):\n                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n                specify the folder name here.\n\n        Returns:\n            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.\n        '
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        token = kwargs.pop('token', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        local_files_only = kwargs.pop('local_files_only', False)
        revision = kwargs.pop('revision', None)
        subfolder = kwargs.pop('subfolder', '')
        from_pipeline = kwargs.pop('_from_pipeline', None)
        from_auto_class = kwargs.pop('_from_auto', False)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        user_agent = {'file_type': 'image processor', 'from_auto_class': from_auto_class}
        if from_pipeline is not None:
            user_agent['using_pipeline'] = from_pipeline
        if is_offline_mode() and (not local_files_only):
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            image_processor_file = os.path.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_image_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            image_processor_file = pretrained_model_name_or_path
            resolved_image_processor_file = download_url(pretrained_model_name_or_path)
        else:
            image_processor_file = IMAGE_PROCESSOR_NAME
            try:
                resolved_image_processor_file = cached_file(pretrained_model_name_or_path, image_processor_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder)
            except EnvironmentError:
                raise
            except Exception:
                raise EnvironmentError(f"Can't load image processor for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {IMAGE_PROCESSOR_NAME} file")
        try:
            with open(resolved_image_processor_file, 'r', encoding='utf-8') as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)
        except json.JSONDecodeError:
            raise EnvironmentError(f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file.")
        if is_local:
            logger.info(f'loading configuration file {resolved_image_processor_file}')
        else:
            logger.info(f'loading configuration file {image_processor_file} from cache at {resolved_image_processor_file}')
        if 'auto_map' in image_processor_dict and (not is_local):
            image_processor_dict['auto_map'] = add_model_info_to_auto_map(image_processor_dict['auto_map'], pretrained_model_name_or_path)
        return (image_processor_dict, kwargs)

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Instantiates a type of [`~image_processing_utils.ImageProcessingMixin`] from a Python dictionary of parameters.\n\n        Args:\n            image_processor_dict (`Dict[str, Any]`):\n                Dictionary that will be used to instantiate the image processor object. Such a dictionary can be\n                retrieved from a pretrained checkpoint by leveraging the\n                [`~image_processing_utils.ImageProcessingMixin.to_dict`] method.\n            kwargs (`Dict[str, Any]`):\n                Additional parameters from which to initialize the image processor object.\n\n        Returns:\n            [`~image_processing_utils.ImageProcessingMixin`]: The image processor object instantiated from those\n            parameters.\n        '
        image_processor_dict = image_processor_dict.copy()
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        if 'size' in kwargs and 'size' in image_processor_dict:
            image_processor_dict['size'] = kwargs.pop('size')
        if 'crop_size' in kwargs and 'crop_size' in image_processor_dict:
            image_processor_dict['crop_size'] = kwargs.pop('crop_size')
        image_processor = cls(**image_processor_dict)
        to_remove = []
        for (key, value) in kwargs.items():
            if hasattr(image_processor, key):
                setattr(image_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info(f'Image processor {image_processor}')
        if return_unused_kwargs:
            return (image_processor, kwargs)
        else:
            return image_processor

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Serializes this instance to a Python dictionary.\n\n        Returns:\n            `Dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.\n        '
        output = copy.deepcopy(self.__dict__)
        output['image_processor_type'] = self.__class__.__name__
        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        if False:
            while True:
                i = 10
        '\n        Instantiates a image processor of type [`~image_processing_utils.ImageProcessingMixin`] from the path to a JSON\n        file of parameters.\n\n        Args:\n            json_file (`str` or `os.PathLike`):\n                Path to the JSON file containing the parameters.\n\n        Returns:\n            A image processor of type [`~image_processing_utils.ImageProcessingMixin`]: The image_processor object\n            instantiated from that JSON file.\n        '
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        image_processor_dict = json.loads(text)
        return cls(**image_processor_dict)

    def to_json_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Serializes this instance to a JSON string.\n\n        Returns:\n            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.\n        '
        dictionary = self.to_dict()
        for (key, value) in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()
        _processor_class = dictionary.pop('_processor_class', None)
        if _processor_class is not None:
            dictionary['processor_class'] = _processor_class
        return json.dumps(dictionary, indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        if False:
            return 10
        "\n        Save this instance to a JSON file.\n\n        Args:\n            json_file_path (`str` or `os.PathLike`):\n                Path to the JSON file in which this image_processor instance's parameters will be saved.\n        "
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__} {self.to_json_string()}'

    @classmethod
    def register_for_auto_class(cls, auto_class='AutoImageProcessor'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register this class with a given auto class. This should only be used for custom image processors as the ones\n        in the library are already mapped with `AutoImageProcessor `.\n\n        <Tip warning={true}>\n\n        This API is experimental and may have some slight breaking changes in the next releases.\n\n        </Tip>\n\n        Args:\n            auto_class (`str` or `type`, *optional*, defaults to `"AutoImageProcessor "`):\n                The auto class to register this new image processor with.\n        '
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module
        if not hasattr(auto_module, auto_class):
            raise ValueError(f'{auto_class} is not a valid auto class.')
        cls._auto_class = auto_class

    def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
        if False:
            while True:
                i = 10
        '\n        Convert a single or a list of urls into the corresponding `PIL.Image` objects.\n\n        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is\n        returned.\n        '
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        elif isinstance(image_url_or_urls, str):
            response = requests.get(image_url_or_urls, stream=True, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            raise ValueError(f'only a single or a list of entries is supported but got type={type(image_url_or_urls)}')

class BaseImageProcessor(ImageProcessingMixin):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        'Preprocess an image or a batch of images.'
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Each image processor must implement its own preprocess method')

    def rescale(self, image: np.ndarray, scale: float, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Rescale an image by a scale factor. image = image * scale.\n\n        Args:\n            image (`np.ndarray`):\n                Image to rescale.\n            scale (`float`):\n                The scaling factor to rescale pixel values by.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n\n        Returns:\n            `np.ndarray`: The rescaled image.\n        '
        return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def normalize(self, image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Normalize an image. image = (image - image_mean) / image_std.\n\n        Args:\n            image (`np.ndarray`):\n                Image to normalize.\n            mean (`float` or `Iterable[float]`):\n                Image mean to use for normalization.\n            std (`float` or `Iterable[float]`):\n                Image standard deviation to use for normalization.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n\n        Returns:\n            `np.ndarray`: The normalized image.\n        '
        return normalize(image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def center_crop(self, image: np.ndarray, size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along\n        any edge, the image is padded with 0\'s and then center cropped.\n\n        Args:\n            image (`np.ndarray`):\n                Image to center crop.\n            size (`Dict[str, int]`):\n                Size of the output image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n        '
        size = get_size_dict(size)
        if 'height' not in size or 'width' not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return center_crop(image, size=(size['height'], size['width']), data_format=data_format, input_data_format=input_data_format, **kwargs)
VALID_SIZE_DICT_KEYS = ({'height', 'width'}, {'shortest_edge'}, {'shortest_edge', 'longest_edge'}, {'longest_edge'})

def is_valid_size_dict(size_dict):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(size_dict, dict):
        return False
    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False

def convert_to_size_dict(size, max_size: Optional[int]=None, default_to_square: bool=True, height_width_order: bool=True):
    if False:
        print('Hello World!')
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError('Cannot specify both size as an int, with default_to_square=True and max_size')
        return {'height': size, 'width': size}
    elif isinstance(size, int) and (not default_to_square):
        size_dict = {'shortest_edge': size}
        if max_size is not None:
            size_dict['longest_edge'] = max_size
        return size_dict
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {'height': size[0], 'width': size[1]}
    elif isinstance(size, (tuple, list)) and (not height_width_order):
        return {'height': size[1], 'width': size[0]}
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError('Cannot specify both default_to_square=True and max_size')
        return {'longest_edge': max_size}
    raise ValueError(f'Could not convert size input to size dict: {size}')

def get_size_dict(size: Union[int, Iterable[int], Dict[str, int]]=None, max_size: Optional[int]=None, height_width_order: bool=True, default_to_square: bool=True, param_name='size') -> dict:
    if False:
        i = 10
        return i + 15
    '\n    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards\n    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,\n    width) or (width, height) format.\n\n    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":\n    size[0]}` if `height_width_order` is `False`.\n    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.\n    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`\n      is set, it is added to the dict as `{"longest_edge": max_size}`.\n\n    Args:\n        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):\n            The `size` parameter to be cast into a size dictionary.\n        max_size (`Optional[int]`, *optional*):\n            The `max_size` parameter to be cast into a size dictionary.\n        height_width_order (`bool`, *optional*, defaults to `True`):\n            If `size` is a tuple, whether it\'s in (height, width) or (width, height) order.\n        default_to_square (`bool`, *optional*, defaults to `True`):\n            If `size` is an int, whether to default to a square image or not.\n    '
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        logger.info(f'{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}. Converted to {size_dict}.')
    else:
        size_dict = size
    if not is_valid_size_dict(size_dict):
        raise ValueError(f'{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}')
    return size_dict
ImageProcessingMixin.push_to_hub = copy_func(ImageProcessingMixin.push_to_hub)
if ImageProcessingMixin.push_to_hub.__doc__ is not None:
    ImageProcessingMixin.push_to_hub.__doc__ = ImageProcessingMixin.push_to_hub.__doc__.format(object='image processor', object_class='AutoImageProcessor', object_files='image processor file')