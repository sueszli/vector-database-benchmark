import json
import sys

def to_markdown(data):
    if False:
        i = 10
        return i + 15
    markdown = ''
    for (key, value) in data.items():
        markdown += f'**{key}:**\n\n'
        if isinstance(value, dict):
            markdown += '| Key | Value |\n| --- | --- |\n'
            for (nested_key, nested_value) in value.items():
                nested_value = round(nested_value, 3) if isinstance(nested_value, float) else {k: round(v, 3) for (k, v) in nested_value.items()} if isinstance(nested_value, dict) else nested_value
                markdown += f'| {nested_key} | {nested_value} |\n'
        elif isinstance(value, list) and all((isinstance(item, dict) for item in value)):
            if value:
                headers = sorted(set().union(*[item.keys() for item in value]))
                markdown += '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
                for item in value:
                    value_list = ['{:.3e}'.format(float(item.get(header, ''))) if not str(item.get(header, '')).isdigit() else str(item.get(header, '')) for header in headers]
                    markdown += '| ' + ' | '.join(value_list) + ' |\n'
            else:
                markdown += '(empty list)\n'
        else:
            markdown += f'{value}\n'
        markdown += '\n'
    return markdown

def json_to_markdown(json_fp, md_fp):
    if False:
        i = 10
        return i + 15
    'Convert a json file to markdown.'
    with open(json_fp, 'r') as file:
        data = json.load(file)
    markdown = to_markdown(data)
    with open(md_fp, 'w') as file:
        file.write(markdown)
    return markdown
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python script.py <json_file> <output_file>')
        sys.exit(1)
    json_file = sys.argv[1]
    md_file = sys.argv[2]
    json_to_markdown(json_file, md_file)