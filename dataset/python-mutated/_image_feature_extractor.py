from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ._pre_trained_models import _get_cache_dir
from turicreate._deps.minimal_package import _minimal_package_import_check
import turicreate.toolkits._tf_utils as _utils

def _create_feature_extractor(model_name):
    if False:
        return 10
    from platform import system
    from ._internal_utils import _mac_ver
    from ._pre_trained_models import IMAGE_MODELS
    from turicreate import extensions
    if system() != 'Darwin' or _mac_ver() < (10, 13):
        ptModel = IMAGE_MODELS[model_name]()
        return TensorFlowFeatureExtractor(ptModel)
    download_path = _get_cache_dir()
    result = extensions.__dict__['image_deep_feature_extractor']()
    result.init_options({'model_name': model_name, 'download_path': download_path})
    return result

class ImageFeatureExtractor(object):

    def __init__(self, ptModel):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        ptModel: ImageClassifierPreTrainedModel\n            An instance of a pre-trained model.\n        '
        pass

    def extract_features(self, dataset, feature):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        dataset: SFrame\n            SFrame with data to extract features from\n        feature: str\n            Name of the column in `dataset` containing the features\n        '
        pass

    def get_coreml_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        model:\n            Return the underlying model in Core ML format\n        '
        pass

class TensorFlowFeatureExtractor(ImageFeatureExtractor):

    def __init__(self, ptModel):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        ptModel: ImageClassifierPreTrainedModel\n            An instance of a pre-trained model.\n        '
        _utils.suppress_tensorflow_warnings()
        keras = _minimal_package_import_check('tensorflow.keras')
        self.gpu_policy = _utils.TensorFlowGPUPolicy()
        self.gpu_policy.start()
        self.ptModel = ptModel
        self.input_shape = ptModel.input_image_shape
        self.coreml_data_layer = ptModel.coreml_data_layer
        self.coreml_feature_layer = ptModel.coreml_feature_layer
        model_path = ptModel.get_model_path('tensorflow')
        self.model = keras.models.load_model(model_path)

    def __del__(self):
        if False:
            return 10
        self.gpu_policy.stop()

    def extract_features(self, dataset, feature, batch_size=64, verbose=False):
        if False:
            print('Hello World!')
        from array import array
        import turicreate as tc
        import numpy as np
        image_sf = tc.SFrame({'image': dataset[feature]})
        state = {}
        state['num_started'] = 0
        state['num_processed'] = 0
        state['total'] = len(dataset)
        state['out'] = tc.SArray(dtype=array)
        if verbose:
            print('Performing feature extraction on resized images...')

        def has_next_batch():
            if False:
                i = 10
                return i + 15
            return state['num_started'] < state['total']

        def next_batch():
            if False:
                while True:
                    i = 10
            start_index = state['num_started']
            end_index = min(start_index + batch_size, state['total'])
            state['num_started'] = end_index
            num_images = end_index - start_index
            shape = (num_images,) + self.ptModel.input_image_shape
            batch = np.zeros(shape, dtype=np.float32)
            tc.extensions.sframe_load_to_numpy(image_sf, batch.ctypes.data, batch.strides, batch.shape, start_index, end_index)
            batch = batch.transpose(0, 2, 3, 1)
            if self.ptModel.input_is_BGR:
                batch = batch[:, :, :, ::-1]
            return batch

        def handle_request(batch):
            if False:
                i = 10
                return i + 15
            y = self.model.predict(batch)
            tf_out = [i.flatten() for i in y]
            return tf_out

        def consume_response(tf_out):
            if False:
                print('Hello World!')
            sa = tc.SArray(tf_out, dtype=array)
            state['out'] = state['out'].append(sa)
            state['num_processed'] += len(tf_out)
            if verbose:
                print('Completed {num_processed:{width}d}/{total:{width}d}'.format(width=len(str(state['total'])), **state))
        while has_next_batch():
            images_in_numpy = next_batch()
            predictions_from_tf = handle_request(images_in_numpy)
            consume_response(predictions_from_tf)
        return state['out']

    def get_coreml_model(self):
        if False:
            while True:
                i = 10
        coremltools = _minimal_package_import_check('coremltools')
        model_path = self.ptModel.get_model_path('coreml')
        return coremltools.models.MLModel(model_path)