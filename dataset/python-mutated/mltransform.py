def mltransform_scale_to_0_1(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    from apache_beam.ml.transforms.base import MLTransform
    from apache_beam.ml.transforms.tft import ScaleTo01
    import tempfile
    data = [{'x': [1, 5, 3]}, {'x': [4, 2, 8]}]
    artifact_location = tempfile.mkdtemp()
    scale_to_0_1_fn = ScaleTo01(columns=['x'])
    with beam.Pipeline() as p:
        transformed_data = p | beam.Create(data) | MLTransform(write_artifact_location=artifact_location).with_transform(scale_to_0_1_fn) | beam.Map(print)
        if test:
            test(transformed_data)

def mltransform_compute_and_apply_vocabulary(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    from apache_beam.ml.transforms.base import MLTransform
    from apache_beam.ml.transforms.tft import ComputeAndApplyVocabulary
    import tempfile
    artifact_location = tempfile.mkdtemp()
    data = [{'x': ['I', 'love', 'Beam']}, {'x': ['Beam', 'is', 'awesome']}]
    compute_and_apply_vocabulary_fn = ComputeAndApplyVocabulary(columns=['x'])
    with beam.Pipeline() as p:
        transformed_data = p | beam.Create(data) | MLTransform(write_artifact_location=artifact_location).with_transform(compute_and_apply_vocabulary_fn) | beam.Map(print)
        if test:
            test(transformed_data)

def mltransform_compute_and_apply_vocabulary_with_scalar(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    from apache_beam.ml.transforms.base import MLTransform
    from apache_beam.ml.transforms.tft import ComputeAndApplyVocabulary
    import tempfile
    data = [{'x': 'I'}, {'x': 'love'}, {'x': 'Beam'}, {'x': 'Beam'}, {'x': 'is'}, {'x': 'awesome'}]
    artifact_location = tempfile.mkdtemp()
    compute_and_apply_vocabulary_fn = ComputeAndApplyVocabulary(columns=['x'])
    with beam.Pipeline() as p:
        transformed_data = p | beam.Create(data) | MLTransform(write_artifact_location=artifact_location).with_transform(compute_and_apply_vocabulary_fn) | beam.Map(print)
        if test:
            test(transformed_data)