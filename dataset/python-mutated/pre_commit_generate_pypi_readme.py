from __future__ import annotations
import re
from pathlib import Path
AIRFLOW_SOURCES = Path(__file__).parents[3].resolve()
README_SECTIONS_TO_EXTRACT = ['Apache Airflow', 'Requirements', 'Getting started', 'Installing from PyPI', 'Official source code', 'Contributing', 'Who uses Apache Airflow', 'Who maintains Apache Airflow']
PYPI_README_HEADER = '<!-- PLEASE DO NOT MODIFY THIS FILE. IT HAS BEEN GENERATED AUTOMATICALLY FROM THE `README.md` FILE OF THE\nPROJECT BY THE `generate-pypi-readme` PRE-COMMIT. YOUR CHANGES HERE WILL BE AUTOMATICALLY OVERWRITTEN.-->\n'

def extract_section(content, section_name):
    if False:
        i = 10
        return i + 15
    start_comment = f'<!-- START {section_name}, please keep comment here to allow auto update of PyPI readme.md -->'
    end_comment = f'<!-- END {section_name}, please keep comment here to allow auto update of PyPI readme.md -->'
    section_match = re.search(f'{re.escape(start_comment)}(.*?)\\n{re.escape(end_comment)}', content, re.DOTALL)
    if section_match:
        return section_match.group(1)
    else:
        raise Exception(f'Cannot find section {section_name} in README.md')
if __name__ == '__main__':
    readme_file = AIRFLOW_SOURCES / 'README.md'
    pypi_readme_file = AIRFLOW_SOURCES / 'generated' / 'PYPI_README.md'
    license_file = AIRFLOW_SOURCES / 'scripts' / 'ci' / 'license-templates' / 'LICENSE.md'
    readme_content = readme_file.read_text()
    generated_pypi_readme_content = license_file.read_text() + '\n' + PYPI_README_HEADER
    for section in README_SECTIONS_TO_EXTRACT:
        section_content = extract_section(readme_content, section)
        generated_pypi_readme_content += section_content
    with pypi_readme_file.open('w') as generated_readme:
        generated_readme.write(generated_pypi_readme_content)