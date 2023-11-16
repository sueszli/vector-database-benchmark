import re
import subprocess
import sys
from pathlib import Path
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command')
SCRIPT_DIR = Path(__file__).resolve().parent
LICENSE_TEXT = (SCRIPT_DIR / 'license-template.txt').read_text().splitlines()[0]
IGNORE_PATTERN = re.compile('^\\.(github|circleci)/|\\.(?:png|jpg|jpeg|gif|ttf|woff|otf|eot|woff2|ico|svg)$|(LICENSE|NOTICES|CODE_OF_CONDUCT\\.md|README\\.md|CONTRIBUTING\\.md|SECURITY.md)$|\\.(json|prettierrc|nvmrc)$|yarn\\.lock$|py\\.typed$|^(\\.dockerignore|\\.editorconfig|\\.gitignore|\\.gitmodules)$|^frontend/(\\.dockerignore|\\.eslintrc.js|\\.prettierignore)$|^lib/(\\.coveragerc|\\.dockerignore|MANIFEST\\.in|mypy\\.ini|pytest\\.ini)$|^lib/(test|dev)-requirements\\.txt$|^lib/min-constraints-gen\\.txt|\\.isort\\.cfg$|\\.credentials/\\.gitignore$|/(fixtures|__snapshots__|test_data|data)/|/vendor/|^vendor/|^component-lib/declarations/apache-arrow|proto/streamlit/proto/openmetrics_data_model\\.proto|^e2e_flaky/scripts/.*\\.py', re.IGNORECASE)

def main():
    if False:
        i = 10
        return i + 15
    git_files = sorted(subprocess.check_output(['git', 'ls-files', '--no-empty-directory']).decode().strip().splitlines())
    invalid_files_count = 0
    for fileloc in git_files:
        if IGNORE_PATTERN.search(fileloc):
            continue
        filepath = Path(fileloc)
        if not filepath.is_file():
            continue
        try:
            file_content = filepath.read_text()
            if LICENSE_TEXT not in file_content:
                print('Found file without license header', fileloc)
                invalid_files_count += 1
        except:
            print(f'Failed to open the file: {fileloc}. Is it binary file?')
            invalid_files_count += 1
    print('Invalid files count:', invalid_files_count)
    if invalid_files_count > 0:
        sys.exit(1)
main()