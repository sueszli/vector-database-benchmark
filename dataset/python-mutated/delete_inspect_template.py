"""Sample app that sets up Data Loss Prevention API inspect templates."""
import argparse
import google.cloud.dlp

def delete_inspect_template(project: str, template_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes a Data Loss Prevention API template.\n    Args:\n        project: The id of the Google Cloud project which owns the template.\n        template_id: The id of the template to delete.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}'
    template_resource = f'{parent}/inspectTemplates/{template_id}'
    dlp.delete_inspect_template(request={'name': template_resource})
    print(f'Template {template_resource} successfully deleted.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template_id', help='The id of the template to delete.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    delete_inspect_template(args.project, args.template_id)