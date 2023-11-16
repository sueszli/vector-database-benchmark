import json
import requests
import urllib.parse as urlparse
DDG_BANGS = 'https://duckduckgo.com/bang.v255.js'

def gen_bangs_json(bangs_file: str) -> None:
    if False:
        print('Hello World!')
    'Generates a json file from the DDG bangs list\n\n    Args:\n        bangs_file: The str path to the new DDG bangs json file\n\n    Returns:\n        None\n\n    '
    try:
        r = requests.get(DDG_BANGS)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    data = json.loads(r.text)
    bangs_data = {}
    for row in data:
        bang_command = '!' + row['t']
        bangs_data[bang_command] = {'url': row['u'].replace('{{{s}}}', '{}'), 'suggestion': bang_command + ' (' + row['s'] + ')'}
    json.dump(bangs_data, open(bangs_file, 'w'))
    print('* Finished creating ddg bangs json')

def resolve_bang(query: str, bangs_dict: dict) -> str:
    if False:
        return 10
    'Transform\'s a user\'s query to a bang search, if an operator is found\n\n    Args:\n        query: The search query\n        bangs_dict: The dict of available bang operators, with corresponding\n                    format string search URLs\n                    (i.e. "!w": "https://en.wikipedia.org...?search={}")\n\n    Returns:\n        str: A formatted redirect for a bang search, or an empty str if there\n             wasn\'t a match or didn\'t contain a bang operator\n\n    '
    if '!' not in query:
        return ''
    split_query = query.strip().split(' ')
    operator = [word for word in split_query if word.lower() in bangs_dict]
    if len(operator) == 1:
        operator = operator[0]
        split_query.remove(operator)
        bang_query = ' '.join(split_query).strip()
        bang = bangs_dict.get(operator.lower(), None)
        if bang:
            bang_url = bang['url']
            if bang_query:
                return bang_url.replace('{}', bang_query, 1)
            else:
                parsed_url = urlparse.urlparse(bang_url)
                return f'{parsed_url.scheme}://{parsed_url.netloc}'
    return ''