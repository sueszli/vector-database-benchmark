"""Sample app that uses the Data Loss Prevent API to redact the contents of
an image file."""
import argparse
import google.cloud.dlp

def redact_image_all_text(project: str, filename: str, output_filename: str) -> None:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to redact all text in an image.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        filename: The path to the file to inspect.\n        output_filename: The path to which the redacted image will be written.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    image_redaction_configs = [{'redact_all_text': True}]
    with open(filename, mode='rb') as f:
        byte_item = {'type_': google.cloud.dlp_v2.FileType.IMAGE, 'data': f.read()}
    parent = f'projects/{project}'
    response = dlp.redact_image(request={'parent': parent, 'image_redaction_configs': image_redaction_configs, 'byte_item': byte_item})
    with open(output_filename, mode='wb') as f:
        f.write(response.redacted_image)
    print('Wrote {byte_count} to {filename}'.format(byte_count=len(response.redacted_image), filename=output_filename))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('filename', help='The path to the file to inspect.')
    parser.add_argument('output_filename', help='The path to which the redacted image will be written.')
    args = parser.parse_args()
    redact_image_all_text(args.project, args.filename, args.output_filename)