def sklearn_unkeyed_model_handler(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    import numpy
    from apache_beam.ml.inference.base import RunInference
    from apache_beam.ml.inference.sklearn_inference import ModelFileType
    from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerNumpy
    sklearn_model_filename = 'gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'
    sklearn_model_handler = SklearnModelHandlerNumpy(model_uri=sklearn_model_filename, model_file_type=ModelFileType.PICKLE)
    unkeyed_data = numpy.array([20, 40, 60, 90], dtype=numpy.float32).reshape(-1, 1)
    with beam.Pipeline() as p:
        predictions = p | 'ReadInputs' >> beam.Create(unkeyed_data) | 'RunInferenceSklearn' >> RunInference(model_handler=sklearn_model_handler) | beam.Map(print)
        if test:
            test(predictions)
if __name__ == '__main__':
    sklearn_unkeyed_model_handler()