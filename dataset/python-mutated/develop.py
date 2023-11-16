"""
Development tasks
"""
from __future__ import absolute_import, print_function
from invoke import Collection, task
from invoke.util import cd
from path import Path
import requests
GHERKIN_LANGUAGES_URL = 'https://raw.githubusercontent.com/cucumber/gherkin/main/gherkin-languages.json'

@task(aliases=['update-languages'])
def update_gherkin(ctx, dry_run=False, verbose=False):
    if False:
        i = 10
        return i + 15
    'Update "gherkin-languages.json" file from cucumber-repo.\n\n    * Download "gherkin-languages.json" from cucumber repo\n    * Update "gherkin-languages.json"\n    * Generate "i18n.py" file from "gherkin-languages.json"\n    * Update "behave/i18n.py" file (optional; not in dry-run mode)\n    '
    with cd('etc/gherkin'):
        gherkin_languages_file = Path('gherkin-languages.json')
        gherkin_languages_file.copy('gherkin-languages.json.SAVED')
        print('Downloading "gherkin-languages.json" from github:cucumber ...')
        download_request = requests.get(GHERKIN_LANGUAGES_URL)
        assert download_request.ok
        print('Download finished: OK (size={0})'.format(len(download_request.content)))
        with open(gherkin_languages_file, 'wb') as f:
            f.write(download_request.content)
        print('Generating "i18n.py" ...')
        ctx.run('./convert_gherkin-languages.py')
        languages_have_changed = False
        result = ctx.run('diff i18n.py ../../behave/i18n.py', warn=True, hide=True)
        languages_have_changed = not result.ok
        if verbose and languages_have_changed:
            print(result.stdout)
        if not languages_have_changed:
            print('NO_CHANGED: gherkin-languages.json')
        elif not dry_run:
            print('Updating behave/i18n.py ...')
            Path('i18n.py').move('../../behave/i18n.py')
namespace = Collection()
namespace.add_task(update_gherkin)
namespace.configure({})