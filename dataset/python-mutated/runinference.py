import torch

class LinearRegression(torch.nn.Module):

    def __init__(self, input_dim=1, output_dim=1):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.linear(x)
        return out

def torch_unkeyed_model_handler(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    import numpy
    import torch
    from apache_beam.ml.inference.base import RunInference
    from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
    model_state_dict_path = 'gs://apache-beam-samples/run_inference/five_times_table_torch.pt'
    model_class = LinearRegression
    model_params = {'input_dim': 1, 'output_dim': 1}
    model_handler = PytorchModelHandlerTensor(model_class=model_class, model_params=model_params, state_dict_path=model_state_dict_path)
    unkeyed_data = numpy.array([10, 40, 60, 90], dtype=numpy.float32).reshape(-1, 1)
    with beam.Pipeline() as p:
        predictions = p | 'InputData' >> beam.Create(unkeyed_data) | 'ConvertNumpyToTensor' >> beam.Map(torch.Tensor) | 'PytorchRunInference' >> RunInference(model_handler=model_handler) | beam.Map(print)
        if test:
            test(predictions)

def torch_keyed_model_handler(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    import torch
    from apache_beam.ml.inference.base import KeyedModelHandler
    from apache_beam.ml.inference.base import RunInference
    from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
    model_state_dict_path = 'gs://apache-beam-samples/run_inference/five_times_table_torch.pt'
    model_class = LinearRegression
    model_params = {'input_dim': 1, 'output_dim': 1}
    keyed_model_handler = KeyedModelHandler(PytorchModelHandlerTensor(model_class=model_class, model_params=model_params, state_dict_path=model_state_dict_path))
    keyed_data = [('first_question', 105.0), ('second_question', 108.0), ('third_question', 1000.0), ('fourth_question', 1013.0)]
    with beam.Pipeline() as p:
        predictions = p | 'KeyedInputData' >> beam.Create(keyed_data) | 'ConvertIntToTensor' >> beam.Map(lambda x: (x[0], torch.Tensor([x[1]]))) | 'PytorchRunInference' >> RunInference(model_handler=keyed_model_handler) | beam.Map(print)
        if test:
            test(predictions)