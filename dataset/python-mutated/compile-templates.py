try:
    from jinja2 import Template
except ImportError:
    raise ImportError('Please install Jinja in order for template generation to succeed')
from pathlib import Path

def find_repo(path):
    if False:
        print('Hello World!')
    for path in Path(path).parents:
        git_dir = path / '.git'
        if git_dir.is_dir():
            return path
repo_root = find_repo(__file__)
roadmap_path = repo_root / 'docs' / 'roadmap.md'
with open(roadmap_path, 'r') as f:
    roadmap_contents_lines = f.readlines()[2:]
    roadmap_contents = ''.join(roadmap_contents_lines)
template_path = repo_root / 'infra' / 'templates' / 'README.md.jinja2'
with open(template_path) as f:
    template = Template(f.read())
readme_md = template.render(roadmap_contents=roadmap_contents)
readme_md = '<!--Do not modify this file. It is auto-generated from a template (infra/templates/README.md.jinja2)-->\n\n' + readme_md
readme_path = repo_root / 'README.md'
with open(readme_path, 'w') as f:
    f.write(readme_md)