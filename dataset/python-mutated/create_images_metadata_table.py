from __future__ import annotations
from collections.abc import Iterable
import io
import json
import logging
import os
import zipfile
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import requests
METADATA_URL = 'https://lilablobssc.blob.core.windows.net/wcs/wcs_camera_traps.json.zip'
INVALID_CATEGORIES = {'#ref!', 'empty', 'end', 'misfire', 'small mammal', 'start', 'unidentifiable', 'unidentified', 'unknown'}

def run(bigquery_dataset: str, bigquery_table: str, pipeline_options: PipelineOptions | None=None) -> None:
    if False:
        while True:
            i = 10
    "Creates the images metadata table in BigQuery.\n\n    This is a one time only process. It reads the metadata file from the LILA\n    science WCS database, gets rid of invalid rows and uploads all the\n    `file_names` alongside their respective `category` into BigQuery.\n\n    To learn more about the WCS Camera Traps dataset:\n        http://lila.science/datasets/wcscameratraps\n\n    Args:\n        bigquery_dataset: Dataset ID for the images database, the dataset must exist.\n        bigquery_table: Table ID for the images database, it is created if it doesn't exist.\n        pipeline_options: PipelineOptions for Apache Beam.\n    "
    schema = ','.join(['category:STRING', 'file_name:STRING'])
    with beam.Pipeline(options=pipeline_options) as pipeline:
        pipeline | 'Create None' >> beam.Create([METADATA_URL]) | 'Get images info' >> beam.FlatMap(get_images_metadata) | 'Filter invalid rows' >> beam.Filter(lambda x: x['category'] not in INVALID_CATEGORIES or x['category'].startswith('unknown ') or x['category'].endswith(' desconocida') or x['category'].endswith(' desconocido')) | 'Write images database' >> beam.io.WriteToBigQuery(dataset=bigquery_dataset, table=bigquery_table, schema=schema, write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE, create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)

def get_images_metadata(metadata_url: str) -> Iterable[dict[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    "Returns an iterable of {'category', 'file_name'} dicts."
    content = requests.get(metadata_url).content
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        filename = os.path.splitext(os.path.basename(metadata_url))[0]
        with zf.open(filename) as f:
            metadata = json.load(f)
    categories = {category['id']: category['name'] for category in metadata['categories']}
    file_names = {image['id']: image['file_name'] for image in metadata['images']}
    for annotation in metadata['annotations']:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        if category_id not in categories:
            logging.error(f'invalid category ID {category_id}, skipping')
        elif image_id not in file_names:
            logging.error(f'invalid image ID {image_id}, skipping')
        else:
            yield {'category': categories[category_id], 'file_name': file_names[image_id]}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bigquery-dataset', required=True, help='BigQuery dataset ID for the images metadata.')
    parser.add_argument('--bigquery-table', default='wildlife_images_metadata', help='BigQuery table ID for the images metadata.')
    (args, pipeline_args) = parser.parse_known_args()
    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True)
    run(args.bigquery_dataset, args.bigquery_table, pipeline_options)