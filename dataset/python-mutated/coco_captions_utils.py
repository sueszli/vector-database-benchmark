"""Implementation of the tfx component functions for the coco captions example."""
import tempfile
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tfx import v1 as tfx

def run_fn(fn_args: tfx.components.FnArgs) -> None:
    if False:
        while True:
            i = 10
    'Build the TF model, train it and export it.'
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=10))
    model.compile()
    model.save(fn_args.serving_model_dir)

def preprocessing_fn(inputs):
    if False:
        while True:
            i = 10
    'Transform raw data.'
    lower = tf.strings.lower(inputs['caption'])
    mean_length = tft.mean(tf.strings.length(lower))
    return {'caption_lower': lower}
if __name__ == '__main__':
    raw_data = [{'caption': 'A bicycle replica with a clock as the front wheel.'}, {'caption': 'A black Honda motorcycle parked in front of a garage.'}, {'caption': 'A room with blue walls and a white sink and door.'}]
    feature_spec = dict(caption=tf.io.FixedLenFeature([], tf.string))
    raw_data_metadata = tft.DatasetMetadata.from_feature_spec(feature_spec)
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        (transformed_dataset, transform_fn) = (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
    (transformed_data, transformed_metadata) = transformed_dataset