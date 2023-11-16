import os
import subprocess
from glob import glob
import hypothesistooling as tools
import hypothesistooling.installers as install
import hypothesistooling.projects.conjecturerust as cr
import hypothesistooling.releasemanagement as rm
from hypothesistooling.junkdrawer import in_dir, once
PACKAGE_NAME = 'hypothesis-ruby'
HYPOTHESIS_RUBY = os.path.join(tools.ROOT, PACKAGE_NAME)
BASE_DIR = HYPOTHESIS_RUBY
TAG_PREFIX = PACKAGE_NAME + '-'
RELEASE_FILE = os.path.join(BASE_DIR, 'RELEASE.md')
CHANGELOG_FILE = os.path.join(BASE_DIR, 'CHANGELOG.md')
GEMSPEC_FILE = os.path.join(BASE_DIR, 'hypothesis-specs.gemspec')
CARGO_FILE = os.path.join(BASE_DIR, 'Cargo.toml')
GEMFILE_LOCK_FILE = os.path.join(BASE_DIR, 'Gemfile.lock')
CONJECTURE_CARGO_FILE = cr.CARGO_FILE
RUST_SRC = cr.SRC
RUBY_SRC = os.path.join(BASE_DIR, 'lib')

def has_release():
    if False:
        for i in range(10):
            print('nop')
    'Is there a version of this package ready to release?'
    return os.path.exists(RELEASE_FILE)

def parse_release_file():
    if False:
        return 10
    return rm.parse_release_file(RELEASE_FILE)

def update_changelog_and_version():
    if False:
        while True:
            i = 10
    'Update the changelog and version based on the current release file.'
    (release_type, release_contents) = parse_release_file()
    version = current_version()
    version_info = rm.parse_version(version)
    (version, version_info) = rm.bump_version_info(version_info, release_type)
    rm.replace_assignment(GEMSPEC_FILE, 's.version', repr(version))
    rm.replace_assignment(GEMSPEC_FILE, 's.date', repr(rm.release_date_string()))
    rm.update_markdown_changelog(CHANGELOG_FILE, name='Hypothesis for Ruby', version=version, entry=release_contents)
    os.unlink(RELEASE_FILE)
LOCAL_PATH_DEPENDENCY = "{ path = '../conjecture-rust' }"

def update_conjecture_dependency(dependency):
    if False:
        i = 10
        return i + 15
    rm.replace_assignment(CARGO_FILE, 'conjecture', dependency)

def build_distribution():
    if False:
        for i in range(10):
            print('nop')
    'Build the rubygem.'
    current_dependency = rm.extract_assignment(CARGO_FILE, 'conjecture')
    assert current_dependency == LOCAL_PATH_DEPENDENCY, f'Cargo file in a bad state. Expected conjecture dependency to be {LOCAL_PATH_DEPENDENCY} but it was instead {current_dependency}'
    conjecture_version = cr.current_version()
    try:
        update_conjecture_dependency(repr(conjecture_version))
        rake_task('gem')
    finally:
        update_conjecture_dependency(LOCAL_PATH_DEPENDENCY)

def tag_name():
    if False:
        print('Hello World!')
    'The tag name for the upcoming release.'
    return TAG_PREFIX + current_version()

def has_source_changes():
    if False:
        i = 10
        return i + 15
    'Returns True if any source files have changed.'
    return tools.has_changes([RUST_SRC, RUBY_SRC, GEMSPEC_FILE]) or cr.has_release()

def current_version():
    if False:
        while True:
            i = 10
    'Returns the current version as specified by the gemspec.'
    ensure_bundler()
    return subprocess.check_output([install.BUNDLER_EXECUTABLE, 'exec', 'ruby', '-e', RUBY_TO_PRINT_VERSION]).decode('ascii').strip()

def bundle(*args):
    if False:
        return 10
    ensure_bundler()
    bundle_command(*args)

def bundle_command(*args):
    if False:
        for i in range(10):
            print('nop')
    with in_dir(BASE_DIR):
        subprocess.check_call([install.BUNDLER_EXECUTABLE, *args])

def rake_task(*args):
    if False:
        for i in range(10):
            print('nop')
    bundle('exec', 'rake', *args)

@once
def ensure_bundler():
    if False:
        for i in range(10):
            print('nop')
    install.ensure_rustup()
    install.ensure_ruby()
    bundle_command('install')

def cargo(*args):
    if False:
        while True:
            i = 10
    install.ensure_rustup()
    with in_dir(BASE_DIR):
        subprocess.check_call(('cargo', *args))
RUBY_TO_PRINT_VERSION = f"\nrequire 'rubygems'\nspec = Gem::Specification::load({GEMSPEC_FILE!r})\nputs spec.version\n".strip().replace('\n', '; ')
RUBYGEMS_CREDENTIALS = os.path.expanduser('~/.gem/credentials')

def upload_distribution():
    if False:
        i = 10
        return i + 15
    'Upload the built package to rubygems.'
    tools.assert_can_release()
    subprocess.check_call([install.GEM_EXECUTABLE, 'push', *glob('hypothesis-specs-*.gem')])