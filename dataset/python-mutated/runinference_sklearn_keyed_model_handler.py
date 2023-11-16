def sklearn_keyed_model_handler(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    from apache_beam.ml.inference.base import KeyedModelHandler
    from apache_beam.ml.inference.base import RunInference
    from apache_beam.ml.inference.sklearn_inference import ModelFileType
    from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerNumpy
    sklearn_model_filename = 'gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'
    sklearn_model_handler = KeyedModelHandler(SklearnModelHandlerNumpy(model_uri=sklearn_model_filename, model_file_type=ModelFileType.PICKLE))
    keyed_data = [('first_question', 105.0), ('second_question', 108.0), ('third_question', 1000.0), ('fourth_question', 1013.0)]
    with beam.Pipeline() as p:
        predictions = p | 'ReadInputs' >> beam.Create(keyed_data) | 'ConvertDataToList' >> beam.Map(lambda x: (x[0], [x[1]])) | 'RunInferenceSklearn' >> RunInference(model_handler=sklearn_model_handler) | beam.Map(print)
        if test:
            test(predictions)
if __name__ == '__main__':
    sklearn_keyed_model_handler()