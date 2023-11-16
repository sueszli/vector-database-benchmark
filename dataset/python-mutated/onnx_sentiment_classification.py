""""A pipeline that uses RunInference to perform sentiment classification
using RoBERTa.

This pipeline takes sentences from a custom text file, and then uses RoBERTa
from Hugging Face to predict the sentiment of a given review. The pipeline
then writes the prediction to an output file in which users can then compare against true labels.

Model is fine-tuned RoBERTa from
https://github.com/SeldonIO/seldon-models/blob/master/pytorch/moviesentiment_roberta/pytorch-roberta-onnx.ipynb # pylint: disable=line-too-long
"""
import argparse
import logging
from typing import Iterable
from typing import Iterator
from typing import Tuple
import numpy as np
import apache_beam as beam
import torch
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.onnx_inference import OnnxModelHandlerNumpy
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from transformers import RobertaTokenizer

def tokenize_sentence(text: str, tokenizer: RobertaTokenizer) -> Tuple[str, torch.Tensor]:
    if False:
        print('Hello World!')
    tokenized_sentence = tokenizer.encode(text, add_special_tokens=True)
    return (text, torch.tensor(tokenized_sentence).numpy())

def filter_empty_lines(text: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    if len(text.strip()) > 0:
        yield text

class PostProcessor(beam.DoFn):

    def process(self, element: Tuple[str, PredictionResult]) -> Iterable[str]:
        if False:
            return 10
        (filename, prediction_result) = element
        prediction = np.argmax(prediction_result.inference, axis=0)
        yield (filename + ';' + str(prediction))

def parse_known_args(argv):
    if False:
        for i in range(10):
            print('nop')
    'Parses args for the workflow.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', help='Path to the text file containing sentences.')
    parser.add_argument('--output', dest='output', required=True, help='Path of file in which to save the output predictions.')
    parser.add_argument('--model_uri', dest='model_uri', required=True, help="Path to the model's uri.")
    return parser.parse_known_args(argv)

def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    if False:
        while True:
            i = 10
    '\n  Args:\n    argv: Command line arguments defined for this example.\n    save_main_session: Used for internal testing.\n    test_pipeline: Used for internal testing.\n  '
    (known_args, pipeline_args) = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    class OnnxNoBatchModelHandler(OnnxModelHandlerNumpy):
        """Wrapper to OnnxModelHandlerNumpy to limit batch size to 1.

    The tokenized strings generated from RobertaTokenizer may have different
    lengths, which doesn't work with torch.stack() in current RunInference
    implementation since stack() requires tensors to be the same size.

    Restricting max_batch_size to 1 means there is only 1 example per `batch`
    in the run_inference() call.
    """

        def batch_elements_kwargs(self):
            if False:
                while True:
                    i = 10
            return {'max_batch_size': 1}
    model_handler = OnnxNoBatchModelHandler(model_uri=known_args.model_uri)
    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    text = pipeline | 'ReadSentences' >> beam.io.ReadFromText(known_args.input)
    text_and_tokenized_text_tuple = text | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines) | 'TokenizeSentence' >> beam.Map(lambda x: tokenize_sentence(x, tokenizer))
    output = text_and_tokenized_text_tuple | 'PyTorchRunInference' >> RunInference(KeyedModelHandler(model_handler)) | 'ProcessOutput' >> beam.ParDo(PostProcessor())
    _ = output | 'WriteOutput' >> beam.io.WriteToText(known_args.output, shard_name_template='', append_trailing_newlines=True)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()