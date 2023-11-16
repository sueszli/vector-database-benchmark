import urllib.request
import flask
import functions_framework
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
_PROJECT_ID = 'YOUR_PROJECT_ID'
_LOCATION = 'us'
_PROCESSOR_ID = 'YOUR_PROCESSOR_ID'

@functions_framework.http
def document_ocr(request: flask.Request) -> flask.Response:
    if False:
        i = 10
        return i + 15
    'BigQuery remote function to process document using Document AI OCR.\n\n    For complete Document AI use cases:\n    https://cloud.google.com/document-ai/docs/samples/documentai-process-ocr-document\n\n    Args:\n        request: HTTP request from BigQuery\n        https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#input_format\n\n    Returns:\n        HTTP response to BigQuery\n        https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#output_format\n    '
    try:
        client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=f'{_LOCATION}-documentai.googleapis.com'))
        processor_name = client.processor_path(_PROJECT_ID, _LOCATION, _PROCESSOR_ID)
        calls = request.get_json()['calls']
        replies = []
        for call in calls:
            content = urllib.request.urlopen(call[0]).read()
            content_type = call[1]
            results = client.process_document({'name': processor_name, 'raw_document': {'content': content, 'mime_type': content_type}})
            replies.append({'text': results.document.text})
        return flask.make_response(flask.jsonify({'replies': replies}))
    except Exception as e:
        return flask.make_response(flask.jsonify({'errorMessage': str(e)}), 400)