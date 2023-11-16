import argparse
import glob
import os
import shutil
import sys
from typing import List
ZULIP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ZULIP_PATH)
import pygments
from scripts.lib import clean_unused_caches
from scripts.lib.zulip_tools import ENDC, OKBLUE, get_dev_uuid_var_path, get_tzdata_zi, is_digest_obsolete, run, run_as_root, write_new_digest
from tools.setup.generate_zulip_bots_static_files import generate_zulip_bots_static_files
from version import PROVISION_VERSION
VENV_PATH = '/srv/zulip-py3-venv'
UUID_VAR_PATH = get_dev_uuid_var_path()
with get_tzdata_zi() as f:
    line = f.readline()
    assert line.startswith('# version ')
    timezones_version = line[len('# version '):]

def create_var_directories() -> None:
    if False:
        for i in range(10):
            print('nop')
    var_dir = os.path.join(ZULIP_PATH, 'var')
    sub_dirs = ['coverage', 'log', 'node-coverage', 'test_uploads', 'uploads', 'xunit-test-results']
    for sub_dir in sub_dirs:
        path = os.path.join(var_dir, sub_dir)
        os.makedirs(path, exist_ok=True)

def build_pygments_data_paths() -> List[str]:
    if False:
        return 10
    paths = ['tools/setup/build_pygments_data', 'tools/setup/lang.json']
    return paths

def build_timezones_data_paths() -> List[str]:
    if False:
        print('Hello World!')
    paths = ['tools/setup/build_timezone_values']
    return paths

def build_landing_page_images_paths() -> List[str]:
    if False:
        return 10
    paths = ['tools/setup/generate_landing_page_images.py']
    paths += glob.glob('static/images/landing-page/hello/original/*')
    return paths

def compilemessages_paths() -> List[str]:
    if False:
        i = 10
        return i + 15
    paths = ['zerver/management/commands/compilemessages.py']
    paths += glob.glob('locale/*/LC_MESSAGES/*.po')
    paths += glob.glob('locale/*/translations.json')
    return paths

def configure_rabbitmq_paths() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    paths = ['scripts/setup/configure-rabbitmq']
    return paths

def setup_shell_profile(shell_profile: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    shell_profile_path = os.path.expanduser(shell_profile)

    def write_command(command: str) -> None:
        if False:
            while True:
                i = 10
        if os.path.exists(shell_profile_path):
            with open(shell_profile_path) as shell_profile_file:
                lines = [line.strip() for line in shell_profile_file.readlines()]
            if command not in lines:
                with open(shell_profile_path, 'a+') as shell_profile_file:
                    shell_profile_file.writelines(command + '\n')
        else:
            with open(shell_profile_path, 'w') as shell_profile_file:
                shell_profile_file.writelines(command + '\n')
    source_activate_command = 'source ' + os.path.join(VENV_PATH, 'bin', 'activate')
    write_command(source_activate_command)
    if os.path.exists('/srv/zulip'):
        write_command('cd /srv/zulip')

def setup_bash_profile() -> None:
    if False:
        print('Hello World!')
    'Select a bash profile file to add setup code to.'
    BASH_PROFILES = [os.path.expanduser(p) for p in ('~/.bash_profile', '~/.bash_login', '~/.profile')]

    def clear_old_profile() -> None:
        if False:
            i = 10
            return i + 15
        BASH_PROFILE = BASH_PROFILES[0]
        DOT_PROFILE = BASH_PROFILES[2]
        OLD_PROFILE_TEXT = 'source /srv/zulip-py3-venv/bin/activate\ncd /srv/zulip\n'
        if os.path.exists(DOT_PROFILE):
            try:
                with open(BASH_PROFILE) as f:
                    profile_contents = f.read()
                if profile_contents == OLD_PROFILE_TEXT:
                    os.unlink(BASH_PROFILE)
            except FileNotFoundError:
                pass
    clear_old_profile()
    for candidate_profile in BASH_PROFILES:
        if os.path.exists(candidate_profile):
            setup_shell_profile(candidate_profile)
            break
    else:
        setup_shell_profile(BASH_PROFILES[0])

def need_to_run_build_pygments_data() -> bool:
    if False:
        while True:
            i = 10
    if not os.path.exists('web/generated/pygments_data.json'):
        return True
    return is_digest_obsolete('build_pygments_data_hash', build_pygments_data_paths(), [pygments.__version__])

def need_to_run_build_timezone_data() -> bool:
    if False:
        while True:
            i = 10
    if not os.path.exists('web/generated/timezones.json'):
        return True
    return is_digest_obsolete('build_timezones_data_hash', build_timezones_data_paths(), [timezones_version])

def need_to_regenerate_landing_page_images() -> bool:
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists('static/images/landing-page/hello/generated'):
        return True
    return is_digest_obsolete('landing_page_images_hash', build_landing_page_images_paths())

def need_to_run_compilemessages() -> bool:
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists('locale/language_name_map.json'):
        print('Need to run compilemessages due to missing language_name_map.json')
        return True
    return is_digest_obsolete('last_compilemessages_hash', compilemessages_paths())

def need_to_run_configure_rabbitmq(settings_list: List[str]) -> bool:
    if False:
        print('Hello World!')
    obsolete = is_digest_obsolete('last_configure_rabbitmq_hash', configure_rabbitmq_paths(), settings_list)
    if obsolete:
        return True
    try:
        from zerver.lib.queue import SimpleQueueClient
        SimpleQueueClient()
        return False
    except Exception:
        return True

def main(options: argparse.Namespace) -> int:
    if False:
        for i in range(10):
            print('nop')
    setup_bash_profile()
    setup_shell_profile('~/.zprofile')
    run(['scripts/setup/generate_secrets.py', '--development'])
    create_var_directories()
    run(['tools/setup/emoji/build_emoji'])
    generate_zulip_bots_static_files()
    if options.is_force or need_to_run_build_pygments_data():
        run(['tools/setup/build_pygments_data'])
        write_new_digest('build_pygments_data_hash', build_pygments_data_paths(), [pygments.__version__])
    else:
        print('No need to run `tools/setup/build_pygments_data`.')
    if options.is_force or need_to_run_build_timezone_data():
        run(['tools/setup/build_timezone_values'])
        write_new_digest('build_timezones_data_hash', build_timezones_data_paths(), [timezones_version])
    else:
        print('No need to run `tools/setup/build_timezone_values`.')
    if options.is_force or need_to_regenerate_landing_page_images():
        run(['tools/setup/generate_landing_page_images.py'])
        write_new_digest('landing_page_images_hash', build_landing_page_images_paths())
    if not options.is_build_release_tarball_only:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'zproject.settings')
        import django
        django.setup()
        from django.conf import settings
        from zerver.lib.test_fixtures import DEV_DATABASE, TEST_DATABASE, destroy_leaked_test_databases
        assert settings.RABBITMQ_PASSWORD is not None
        if options.is_force or need_to_run_configure_rabbitmq([settings.RABBITMQ_PASSWORD]):
            run_as_root(['scripts/setup/configure-rabbitmq'])
            write_new_digest('last_configure_rabbitmq_hash', configure_rabbitmq_paths(), [settings.RABBITMQ_PASSWORD])
        else:
            print('No need to run `scripts/setup/configure-rabbitmq.')
        dev_template_db_status = DEV_DATABASE.template_status()
        if options.is_force or dev_template_db_status == 'needs_rebuild':
            run(['tools/setup/postgresql-init-dev-db'])
            if options.skip_dev_db_build:
                pass
            else:
                run(['tools/rebuild-dev-database'])
                DEV_DATABASE.write_new_db_digest()
        elif dev_template_db_status == 'run_migrations':
            DEV_DATABASE.run_db_migrations()
        elif dev_template_db_status == 'current':
            print('No need to regenerate the dev DB.')
        test_template_db_status = TEST_DATABASE.template_status()
        if options.is_force or test_template_db_status == 'needs_rebuild':
            run(['tools/setup/postgresql-init-test-db'])
            run(['tools/rebuild-test-database'])
            TEST_DATABASE.write_new_db_digest()
        elif test_template_db_status == 'run_migrations':
            TEST_DATABASE.run_db_migrations()
        elif test_template_db_status == 'current':
            print('No need to regenerate the test DB.')
        if options.is_force or need_to_run_compilemessages():
            run(['./manage.py', 'compilemessages', '--ignore=*'])
            write_new_digest('last_compilemessages_hash', compilemessages_paths())
        else:
            print('No need to run `manage.py compilemessages`.')
        destroyed = destroy_leaked_test_databases()
        if destroyed:
            print(f'Dropped {destroyed} stale test databases!')
    clean_unused_caches.main(argparse.Namespace(threshold_days=6, dry_run=False, verbose=False, no_headings=True))
    if os.path.isfile('.eslintcache'):
        os.remove('.eslintcache')
    print('Cleaning var/ directory files...')
    var_paths = glob.glob('var/test*')
    var_paths.append('var/bot_avatar')
    for path in var_paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            pass
    version_file = os.path.join(UUID_VAR_PATH, 'provision_version')
    print(f'writing to {version_file}\n')
    with open(version_file, 'w') as f:
        f.write('.'.join(map(str, PROVISION_VERSION)) + '\n')
    print()
    print(OKBLUE + 'Zulip development environment setup succeeded!' + ENDC)
    return 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', dest='is_force', help='Ignore all provisioning optimizations.')
    parser.add_argument('--build-release-tarball-only', action='store_true', dest='is_build_release_tarball_only', help='Provision for test suite with production settings.')
    parser.add_argument('--skip-dev-db-build', action='store_true', help="Don't run migrations on dev database.")
    options = parser.parse_args()
    sys.exit(main(options))