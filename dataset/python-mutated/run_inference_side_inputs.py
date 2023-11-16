"""
Used for internal testing. No backwards compatibility.
"""
import argparse
import logging
import time
from typing import Iterable
from typing import Optional
from typing import Sequence
import apache_beam as beam
from apache_beam.ml.inference import base
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.transforms import trigger
from apache_beam.transforms import window
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.transforms.userstate import CombiningValueStateSpec

class FakeModelDefault:

    def predict(self, example: int) -> int:
        if False:
            while True:
                i = 10
        return example

class FakeModelAdd(FakeModelDefault):

    def predict(self, example: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return example + 1

class FakeModelSub(FakeModelDefault):

    def predict(self, example: int) -> int:
        if False:
            print('Hello World!')
        return example - 1

class FakeModelHandlerReturnsPredictionResult(base.ModelHandler[int, base.PredictionResult, FakeModelDefault]):

    def __init__(self, clock=None, model_id='model_default'):
        if False:
            while True:
                i = 10
        self.model_id = model_id
        self._fake_clock = clock

    def load_model(self):
        if False:
            return 10
        if self._fake_clock:
            self._fake_clock.current_time_ns += 500000000
        if self.model_id == 'model_add.pkl':
            return FakeModelAdd()
        elif self.model_id == 'model_sub.pkl':
            return FakeModelSub()
        return FakeModelDefault()

    def run_inference(self, batch: Sequence[int], model: FakeModelDefault, inference_args=None) -> Iterable[base.PredictionResult]:
        if False:
            while True:
                i = 10
        for example in batch:
            yield base.PredictionResult(model_id=self.model_id, example=example, inference=model.predict(example))

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            print('Hello World!')
        self.model_id = model_path if model_path else self.model_id

def run(argv=None, save_main_session=True):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    first_ts = time.time()
    side_input_interval = 60
    main_input_interval = 20
    last_ts = first_ts + 1200
    mid_ts = (first_ts + last_ts) / 2
    (_, pipeline_args) = parser.parse_known_args(argv)
    options = PipelineOptions(pipeline_args)
    options.view_as(SetupOptions).save_main_session = save_main_session

    class GetModel(beam.DoFn):

        def process(self, element) -> Iterable[base.ModelMetadata]:
            if False:
                while True:
                    i = 10
            if time.time() > mid_ts:
                yield base.ModelMetadata(model_id='model_add.pkl', model_name='model_add')
            else:
                yield base.ModelMetadata(model_id='model_sub.pkl', model_name='model_sub')

    class _EmitSingletonSideInput(beam.DoFn):
        COUNT_STATE = CombiningValueStateSpec('count', combine_fn=sum)

        def process(self, element, count_state=beam.DoFn.StateParam(COUNT_STATE)):
            if False:
                i = 10
                return i + 15
            (_, path) = element
            counter = count_state.read()
            if counter == 0:
                count_state.add(1)
                yield path

    def validate_prediction_result(x: base.PredictionResult):
        if False:
            i = 10
            return i + 15
        model_id = x.model_id
        if model_id == 'model_sub.pkl':
            assert x.example == 1 and x.inference == 0
        if model_id == 'model_add.pkl':
            assert x.example == 1 and x.inference == 2
        if model_id == 'model_default':
            assert x.example == 1 and x.inference == 1
    with beam.Pipeline(options=options) as pipeline:
        side_input = pipeline | 'SideInputPColl' >> PeriodicImpulse(first_ts, last_ts, fire_interval=side_input_interval) | 'GetModelId' >> beam.ParDo(GetModel()) | 'AttachKey' >> beam.Map(lambda x: (x, x)) | 'GetSingleton' >> beam.ParDo(_EmitSingletonSideInput()) | 'ApplySideInputWindow' >> beam.WindowInto(window.GlobalWindows(), trigger=trigger.Repeatedly(trigger.AfterProcessingTime(1)), accumulation_mode=trigger.AccumulationMode.DISCARDING)
        model_handler = FakeModelHandlerReturnsPredictionResult()
        inference_pcoll = pipeline | 'MainInputPColl' >> PeriodicImpulse(first_ts, last_ts, fire_interval=main_input_interval, apply_windowing=True) | beam.Map(lambda x: 1) | base.RunInference(model_handler=model_handler, model_metadata_pcoll=side_input)
        _ = inference_pcoll | 'AssertPredictionResult' >> beam.Map(validate_prediction_result)
        _ = inference_pcoll | 'Logging' >> beam.Map(logging.info)
if __name__ == '__main__':
    run()