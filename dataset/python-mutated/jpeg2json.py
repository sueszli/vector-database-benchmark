"""jpeg2json.py: Converts a JPEG image into a json request to CloudML.

Usage:
python jpeg2json.py 002s_C6_ImagerDefaults_9.jpg > request.json

See:
https://cloud.google.com/ml-engine/docs/concepts/prediction-overview#online_prediction_input_data
"""
import base64
import sys

def to_json(data):
    if False:
        print('Hello World!')
    return '{"image_bytes":{"b64": "%s"}}' % base64.b64encode(data)
if __name__ == '__main__':
    file = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    print(to_json(file.read()))