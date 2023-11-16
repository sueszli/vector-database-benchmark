"""Sample app that uses the Data Loss Prevent API to redact the contents of
an image file."""
import argparse
import google.cloud.dlp

def redact_image_all_info_types(project: str, filename: str, output_filename: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to redact protected data in an image.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        filename: The path to the file to inspect.\n        output_filename: The path to which the redacted image will be written.\n            A full list of info type categories can be fetched from the API.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    with open(filename, mode='rb') as f:
        byte_item = {'type_': google.cloud.dlp_v2.FileType.IMAGE, 'data': f.read()}
    parent = f'projects/{project}'
    response = dlp.redact_image(request={'parent': parent, 'byte_item': byte_item})
    with open(output_filename, mode='wb') as f:
        f.write(response.redacted_image)
    print(f'Wrote {len(response.redacted_image)} to {output_filename}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('filename', help='The path to the file to inspect.')
    parser.add_argument('output_filename', help='The path to which the redacted image will be written.')
    args = parser.parse_args()
    redact_image_all_info_types(args.project, args.filename, args.output_filename)