"""Sample app that queries the Data Loss Prevention API for supported
categories and info types."""
import argparse
from typing import Optional
import google.cloud.dlp

def list_info_types(language_code: Optional[str]=None, result_filter: Optional[str]=None) -> None:
    if False:
        return 10
    'List types of sensitive information within a category.\n    Args:\n        language_code: The BCP-47 language code to use, e.g. \'en-US\'.\n        result_filter: An optional filter to only return info types supported by\n                certain parts of the API. Defaults to "supported_by=INSPECT".\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    response = dlp.list_info_types(request={'parent': language_code, 'filter': result_filter})
    print('Info types:')
    for info_type in response.info_types:
        print('{name}: {display_name}'.format(name=info_type.name, display_name=info_type.display_name))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language_code', help="The BCP-47 language code to use, e.g. 'en-US'.")
    parser.add_argument('--filter', help='An optional filter to only return info types supported by certain parts of the API. Defaults to "supported_by=INSPECT".')
    args = parser.parse_args()
    list_info_types(language_code=args.language_code, result_filter=args.filter)