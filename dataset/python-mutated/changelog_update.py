import re
markdown_text = "\n## What's Changed\n* Add installation CI by @giswqs in https://github.com/gee-community/geemap/pull/1656\n* Fix vis control error by @giswqs in https://github.com/gee-community/geemap/pull/1660\n\n## New Contributors\n* @bengalin made their first contribution in https://github.com/gee-community/geemap/pull/1664\n* @sufyanAbbasi made their first contribution in https://github.com/gee-community/geemap/pull/1666\n* @kirimaru-jp made their first contribution in https://github.com/gee-community/geemap/pull/1669\n* @schwehr made their first contribution in https://github.com/gee-community/geemap/pull/1673\n\n**Full Changelog**: https://github.com/gee-community/geemap/compare/v0.25.0...v0.26.0\n"
pattern = 'https://github\\.com/gee-community/geemap/pull/(\\d+)'

def replace_url(match):
    if False:
        print('Hello World!')
    pr_number = match.group(1)
    return f'[#{pr_number}](https://github.com/gee-community/geemap/pull/{pr_number})'
formatted_text = re.sub(pattern, replace_url, markdown_text)
for line in formatted_text.splitlines():
    if 'Full Changelog' in line:
        prefix = line.split(': ')[0]
        link = line.split(': ')[1]
        version = line.split('/')[-1]
        formatted_text = formatted_text.replace(line, f'{prefix}: [{version}]({link})').replace("## What's Changed", "**What's Changed**").replace('## New Contributors', '**New Contributors**')
with open('docs/changelog_update.md', 'w') as f:
    f.write(formatted_text)
print(formatted_text)