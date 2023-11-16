"""
Processor class for TVLT.
"""
from ...processing_utils import ProcessorMixin

class TvltProcessor(ProcessorMixin):
    """
    Constructs a TVLT processor which wraps a TVLT image processor and TVLT feature extractor into a single processor.

    [`TvltProcessor`] offers all the functionalities of [`TvltImageProcessor`] and [`TvltFeatureExtractor`]. See the
    docstring of [`~TvltProcessor.__call__`] for more information.

    Args:
        image_processor (`TvltImageProcessor`):
            An instance of [`TvltImageProcessor`]. The image processor is a required input.
        feature_extractor (`TvltFeatureExtractor`):
            An instance of [`TvltFeatureExtractor`]. The feature extractor is a required input.
    """
    attributes = ['image_processor', 'feature_extractor']
    image_processor_class = 'TvltImageProcessor'
    feature_extractor_class = 'TvltFeatureExtractor'

    def __init__(self, image_processor, feature_extractor):
        if False:
            print('Hello World!')
        super().__init__(image_processor=image_processor, feature_extractor=feature_extractor)
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor

    def __call__(self, images=None, audio=None, images_mixed=None, sampling_rate=None, mask_audio=False, mask_pixel=False, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        Forwards the `images` argument to TvltImageProcessor's [`~TvltImageProcessor.preprocess`] and the `audio`\n        argument to TvltFeatureExtractor's [`~TvltFeatureExtractor.__call__`]. Please refer to the docstring of the\n        above two methods for more information.\n        "
        if images is None and audio is None:
            raise ValueError('You need to specify either an `images` or `audio` input to process.')
        images_mixed_dict = None
        if images is not None:
            images_dict = self.image_processor(images, *args, mask_pixel=mask_pixel, **kwargs)
        if images_mixed is not None:
            images_mixed_dict = self.image_processor(images_mixed, *args, is_mixed=True, **kwargs)
        if audio is not None:
            audio_dict = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, mask_audio=mask_audio, **kwargs)
        output_dict = {}
        if audio is not None:
            output_dict.update(audio_dict)
        if images is not None:
            output_dict.update(images_dict)
        if images_mixed_dict is not None:
            output_dict.update(images_mixed_dict)
        return output_dict

    @property
    def model_input_names(self):
        if False:
            print('Hello World!')
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + feature_extractor_input_names))