import urllib.request
import flask
import functions_framework
from google.cloud import vision

@functions_framework.http
def label_detection(request: flask.Request) -> flask.Response:
    if False:
        print('Hello World!')
    'BigQuery remote function to label input images.\n    Args:\n        request: HTTP request from BigQuery\n        https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#input_format\n    Returns:\n        HTTP response to BigQuery\n        https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#output_format\n    '
    try:
        client = vision.ImageAnnotatorClient()
        calls = request.get_json()['calls']
        replies = []
        for call in calls:
            content = urllib.request.urlopen(call[0]).read()
            results = client.label_detection({'content': content})
            replies.append(vision.AnnotateImageResponse.to_dict(results))
        return flask.make_response(flask.jsonify({'replies': replies}))
    except Exception as e:
        return flask.make_response(flask.jsonify({'errorMessage': str(e)}), 400)