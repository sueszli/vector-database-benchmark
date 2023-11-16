"""Sample app that sets up Data Loss Prevention API inspect templates."""
import argparse
from typing import List
from typing import Optional
import google.cloud.dlp

def create_inspect_template(project: str, info_types: List[str], template_id: Optional[str]=None, display_name: Optional[str]=None, min_likelihood: Optional[int]=None, max_findings: Optional[int]=None, include_quote: Optional[bool]=None) -> None:
    if False:
        while True:
            i = 10
    "Creates a Data Loss Prevention API inspect template.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        template_id: The id of the template. If omitted, an id will be randomly\n            generated.\n        display_name: The optional display name of the template.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        include_quote: Boolean for whether to display a quote of the detected\n            information in the results.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'min_likelihood': min_likelihood, 'include_quote': include_quote, 'limits': {'max_findings_per_request': max_findings}}
    inspect_template = {'inspect_config': inspect_config, 'display_name': display_name}
    parent = f'projects/{project}'
    response = dlp.create_inspect_template(request={'parent': parent, 'inspect_template': inspect_template, 'template_id': template_id})
    print(f'Successfully created template {response.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_id', help='The id of the template. If omitted, an id will be randomly generated')
    parser.add_argument('--display_name', help='The optional display name of the template.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--include_quote', type=bool, help='A boolean for whether to display a quote of the detected information in the results.', default=True)
    args = parser.parse_args()
    create_inspect_template(args.project, args.info_types, template_id=args.template_id, display_name=args.display_name, min_likelihood=args.min_likelihood, max_findings=args.max_findings, include_quote=args.include_quote)