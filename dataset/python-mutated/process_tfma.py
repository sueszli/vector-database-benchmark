"""Runs a batch job for performing Tensorflow Model Analysis."""
import argparse
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.evaluators import evaluator
import apache_beam as beam
from apache_beam.io.gcp.bigquery import ReadFromBigQuery
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.load_tests.load_test_metrics_utils import MeasureTime
from apache_beam.testing.load_tests.load_test_metrics_utils import MetricsReader
from trainer import taxi

def process_tfma(schema_file, big_query_table=None, eval_model_dir=None, max_eval_rows=None, pipeline_args=None, publish_to_bq=False, project=None, metrics_table=None, metrics_dataset=None):
    if False:
        print('Hello World!')
    'Runs a batch job to evaluate the eval_model against the given input.\n\n  Args:\n  schema_file: A file containing a text-serialized Schema that describes the\n      eval data.\n  big_query_table: A BigQuery table name specified as DATASET.TABLE which\n      should be the input for evaluation. This can only be set if input_csv is\n      None.\n  eval_model_dir: A directory where the eval model is located.\n  max_eval_rows: Number of rows to query from BigQuery.\n  pipeline_args: additional DataflowRunner or DirectRunner args passed to\n  the beam pipeline.\n  publish_to_bq:\n  project:\n  metrics_dataset:\n  metrics_table:\n\n  Raises:\n  ValueError: if input_csv and big_query_table are not specified correctly.\n  '
    if big_query_table is None:
        raise ValueError('--big_query_table should be provided.')
    slice_spec = [tfma.slicer.SingleSliceSpec(), tfma.slicer.SingleSliceSpec(columns=['trip_start_hour'])]
    metrics_namespace = metrics_table
    schema = taxi.read_schema(schema_file)
    eval_shared_model = tfma.default_eval_shared_model(eval_saved_model_path=eval_model_dir, add_metrics_callbacks=[tfma.post_export_metrics.calibration_plot_and_prediction_histogram(), tfma.post_export_metrics.auc_plots()])
    metrics_monitor = None
    if publish_to_bq:
        metrics_monitor = MetricsReader(publish_to_bq=publish_to_bq, project_name=project, bq_table=metrics_table, bq_dataset=metrics_dataset, namespace=metrics_namespace, filters=MetricsFilter().with_namespace(metrics_namespace))
    pipeline = beam.Pipeline(argv=pipeline_args)
    query = taxi.make_sql(big_query_table, max_eval_rows, for_eval=True)
    raw_feature_spec = taxi.get_raw_feature_spec(schema)
    raw_data = pipeline | 'ReadBigQuery' >> ReadFromBigQuery(query=query, project=project, use_standard_sql=True) | 'Measure time: Start' >> beam.ParDo(MeasureTime(metrics_namespace)) | 'CleanData' >> beam.Map(lambda x: taxi.clean_raw_data_dict(x, raw_feature_spec))
    coder = taxi.make_proto_coder(schema)
    extractors = tfma.default_extractors(eval_shared_model=eval_shared_model, slice_spec=slice_spec, desired_batch_size=None, materialize=False)
    evaluators = tfma.default_evaluators(eval_shared_model=eval_shared_model, desired_batch_size=None, num_bootstrap_samples=1)
    _ = raw_data | 'ToSerializedTFExample' >> beam.Map(coder.encode) | 'Extract Results' >> tfma.InputsToExtracts() | 'Extract and evaluate' >> tfma.ExtractAndEvaluate(extractors=extractors, evaluators=evaluators) | 'Map Evaluations to PCollection' >> MapEvalToPCollection() | 'Measure time: End' >> beam.ParDo(MeasureTime(metrics_namespace))
    result = pipeline.run()
    result.wait_until_finish()
    if metrics_monitor:
        metrics_monitor.publish_metrics(result)

@beam.ptransform_fn
@beam.typehints.with_input_types(evaluator.Evaluation)
@beam.typehints.with_output_types(beam.typehints.Any)
def MapEvalToPCollection(evaluation):
    if False:
        while True:
            i = 10
    return evaluation['metrics']

def main():
    if False:
        i = 10
        return i + 15
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model_dir', help='Input path to the model which will be evaluated.')
    parser.add_argument('--big_query_table', help='BigQuery path to input examples which will be evaluated.')
    parser.add_argument('--max_eval_rows', help='Maximum number of rows to evaluate on.', default=None, type=int)
    parser.add_argument('--schema_file', help='File holding the schema for the input data')
    parser.add_argument('--publish_to_big_query', help='Whether to publish to BQ', default=None, type=bool)
    parser.add_argument('--metrics_dataset', help='BQ dataset', default=None, type=str)
    parser.add_argument('--metrics_table', help='BQ table for storing metrics', default=None, type=str)
    parser.add_argument('--metric_reporting_project', help='BQ table project', default=None, type=str)
    (known_args, pipeline_args) = parser.parse_known_args()
    process_tfma(big_query_table=known_args.big_query_table, eval_model_dir=known_args.eval_model_dir, max_eval_rows=known_args.max_eval_rows, schema_file=known_args.schema_file, pipeline_args=pipeline_args, publish_to_bq=known_args.publish_to_big_query, metrics_table=known_args.metrics_table, metrics_dataset=known_args.metrics_dataset, project=known_args.metric_reporting_project)
if __name__ == '__main__':
    main()