from __future__ import annotations
DOCUMENTATION = "\nauthor: 'Ansible Core Team (@ansible)'\nshort_description: 'Add and remove deb822 formatted repositories'\ndescription:\n- 'Add and remove deb822 formatted repositories in Debian based distributions'\nmodule: deb822_repository\nnotes:\n- This module will not automatically update caches, call the apt module based\n  on the changed state.\noptions:\n    allow_downgrade_to_insecure:\n        description:\n        - Allow downgrading a package that was previously authenticated but\n          is no longer authenticated\n        type: bool\n    allow_insecure:\n        description:\n        - Allow insecure repositories\n        type: bool\n    allow_weak:\n        description:\n        - Allow repositories signed with a key using a weak digest algorithm\n        type: bool\n    architectures:\n        description:\n        - 'Architectures to search within repository'\n        type: list\n        elements: str\n    by_hash:\n        description:\n        - Controls if APT should try to acquire indexes via a URI constructed\n          from a hashsum of the expected file instead of using the well-known\n          stable filename of the index.\n        type: bool\n    check_date:\n        description:\n        - Controls if APT should consider the machine's time correct and hence\n          perform time related checks, such as verifying that a Release file\n          is not from the future.\n        type: bool\n    check_valid_until:\n        description:\n        - Controls if APT should try to detect replay attacks.\n        type: bool\n    components:\n        description:\n        - Components specify different sections of one distribution version\n          present in a Suite.\n        type: list\n        elements: str\n    date_max_future:\n        description:\n        - Controls how far from the future a repository may be.\n        type: int\n    enabled:\n        description:\n        - Tells APT whether the source is enabled or not.\n        type: bool\n    inrelease_path:\n        description:\n        - Determines the path to the InRelease file, relative to the normal\n          position of an InRelease file.\n        type: str\n    languages:\n        description:\n        - Defines which languages information such as translated\n          package descriptions should be downloaded.\n        type: list\n        elements: str\n    name:\n        description:\n        - Name of the repo. Specifically used for C(X-Repolib-Name) and in\n          naming the repository and signing key files.\n        required: true\n        type: str\n    pdiffs:\n        description:\n        - Controls if APT should try to use PDiffs to update old indexes\n          instead of downloading the new indexes entirely\n        type: bool\n    signed_by:\n        description:\n        - Either a URL to a GPG key, absolute path to a keyring file, one or\n          more fingerprints of keys either in the C(trusted.gpg) keyring or in\n          the keyrings in the C(trusted.gpg.d/) directory, or an ASCII armored\n          GPG public key block.\n        type: str\n    suites:\n        description:\n        - >-\n          Suite can specify an exact path in relation to the URI(s) provided,\n          in which case the Components: must be omitted and suite must end\n          with a slash (C(/)). Alternatively, it may take the form of a\n          distribution version (e.g. a version codename like disco or artful).\n          If the suite does not specify a path, at least one component must\n          be present.\n        type: list\n        elements: str\n    targets:\n        description:\n        - Defines which download targets apt will try to acquire from this\n          source.\n        type: list\n        elements: str\n    trusted:\n        description:\n        - Decides if a source is considered trusted or if warnings should be\n          raised before e.g. packages are installed from this source.\n        type: bool\n    types:\n        choices:\n        - deb\n        - deb-src\n        default:\n        - deb\n        type: list\n        elements: str\n        description:\n        - Which types of packages to look for from a given source; either\n          binary V(deb) or source code V(deb-src)\n    uris:\n        description:\n        - The URIs must specify the base of the Debian distribution archive,\n          from which APT finds the information it needs.\n        type: list\n        elements: str\n    mode:\n        description:\n        - The octal mode for newly created files in sources.list.d.\n        type: raw\n        default: '0644'\n    state:\n        description:\n        - A source string state.\n        type: str\n        choices:\n        - absent\n        - present\n        default: present\nrequirements:\n    - python3-debian / python-debian\nversion_added: '2.15'\n"
EXAMPLES = "\n- name: Add debian repo\n  deb822_repository:\n    name: debian\n    types: deb\n    uris: http://deb.debian.org/debian\n    suites: stretch\n    components:\n      - main\n      - contrib\n      - non-free\n\n- name: Add debian repo with key\n  deb822_repository:\n    name: debian\n    types: deb\n    uris: https://deb.debian.org\n    suites: stable\n    components:\n      - main\n      - contrib\n      - non-free\n    signed_by: |-\n      -----BEGIN PGP PUBLIC KEY BLOCK-----\n\n      mDMEYCQjIxYJKwYBBAHaRw8BAQdAD/P5Nvvnvk66SxBBHDbhRml9ORg1WV5CvzKY\n      CuMfoIS0BmFiY2RlZoiQBBMWCgA4FiEErCIG1VhKWMWo2yfAREZd5NfO31cFAmAk\n      IyMCGyMFCwkIBwMFFQoJCAsFFgIDAQACHgECF4AACgkQREZd5NfO31fbOwD6ArzS\n      dM0Dkd5h2Ujy1b6KcAaVW9FOa5UNfJ9FFBtjLQEBAJ7UyWD3dZzhvlaAwunsk7DG\n      3bHcln8DMpIJVXht78sL\n      =IE0r\n      -----END PGP PUBLIC KEY BLOCK-----\n\n- name: Add repo using key from URL\n  deb822_repository:\n    name: example\n    types: deb\n    uris: https://download.example.com/linux/ubuntu\n    suites: '{{ ansible_distribution_release }}'\n    components: stable\n    architectures: amd64\n    signed_by: https://download.example.com/linux/ubuntu/gpg\n"
RETURN = '\nrepo:\n  description: A source string for the repository\n  returned: always\n  type: str\n  sample: |\n    X-Repolib-Name: debian\n    Types: deb\n    URIs: https://deb.debian.org\n    Suites: stable\n    Components: main contrib non-free\n    Signed-By:\n        -----BEGIN PGP PUBLIC KEY BLOCK-----\n        .\n        mDMEYCQjIxYJKwYBBAHaRw8BAQdAD/P5Nvvnvk66SxBBHDbhRml9ORg1WV5CvzKY\n        CuMfoIS0BmFiY2RlZoiQBBMWCgA4FiEErCIG1VhKWMWo2yfAREZd5NfO31cFAmAk\n        IyMCGyMFCwkIBwMFFQoJCAsFFgIDAQACHgECF4AACgkQREZd5NfO31fbOwD6ArzS\n        dM0Dkd5h2Ujy1b6KcAaVW9FOa5UNfJ9FFBtjLQEBAJ7UyWD3dZzhvlaAwunsk7DG\n        3bHcln8DMpIJVXht78sL\n        =IE0r\n        -----END PGP PUBLIC KEY BLOCK-----\n\ndest:\n  description: Path to the repository file\n  returned: always\n  type: str\n  sample: /etc/apt/sources.list.d/focal-archive.sources\n\nkey_filename:\n  description: Path to the signed_by key file\n  returned: always\n  type: str\n  sample: /etc/apt/keyrings/debian.gpg\n'
import os
import re
import tempfile
import textwrap
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import raise_from
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import open_url
from ansible.module_utils.urls import get_user_agent
from ansible.module_utils.urls import urlparse
HAS_DEBIAN = True
DEBIAN_IMP_ERR = None
try:
    from debian.deb822 import Deb822
except ImportError:
    HAS_DEBIAN = False
    DEBIAN_IMP_ERR = traceback.format_exc()
KEYRINGS_DIR = '/etc/apt/keyrings'

def ensure_keyrings_dir(module):
    if False:
        while True:
            i = 10
    changed = False
    if not os.path.isdir(KEYRINGS_DIR):
        if not module.check_mode:
            os.mkdir(KEYRINGS_DIR, 493)
        changed |= True
    changed |= module.set_fs_attributes_if_different({'path': KEYRINGS_DIR, 'secontext': [None, None, None], 'owner': 'root', 'group': 'root', 'mode': '0755', 'attributes': None}, changed)
    return changed

def make_signed_by_filename(slug, ext):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(KEYRINGS_DIR, '%s.%s' % (slug, ext))

def make_sources_filename(slug):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join('/etc/apt/sources.list.d', '%s.sources' % slug)

def format_bool(v):
    if False:
        return 10
    return 'yes' if v else 'no'

def format_list(v):
    if False:
        for i in range(10):
            print('nop')
    return ' '.join(v)

def format_multiline(v):
    if False:
        while True:
            i = 10
    return '\n' + textwrap.indent('\n'.join((line.strip() or '.' for line in v.strip().splitlines())), '    ')

def format_field_name(v):
    if False:
        for i in range(10):
            print('nop')
    if v == 'name':
        return 'X-Repolib-Name'
    elif v == 'uris':
        return 'URIs'
    return v.replace('_', '-').title()

def is_armored(b_data):
    if False:
        for i in range(10):
            print('nop')
    return b'-----BEGIN PGP PUBLIC KEY BLOCK-----' in b_data

def write_signed_by_key(module, v, slug):
    if False:
        i = 10
        return i + 15
    changed = False
    if os.path.isfile(v):
        return (changed, v, None)
    b_data = None
    parts = generic_urlparse(urlparse(v))
    if parts.scheme:
        try:
            r = open_url(v, http_agent=get_user_agent())
        except Exception as exc:
            raise_from(RuntimeError(to_native(exc)), exc)
        else:
            b_data = r.read()
    else:
        return (changed, None, v)
    if not b_data:
        return (changed, v, None)
    (tmpfd, tmpfile) = tempfile.mkstemp(dir=module.tmpdir)
    with os.fdopen(tmpfd, 'wb') as f:
        f.write(b_data)
    ext = 'asc' if is_armored(b_data) else 'gpg'
    filename = make_signed_by_filename(slug, ext)
    src_chksum = module.sha256(tmpfile)
    dest_chksum = module.sha256(filename)
    if src_chksum != dest_chksum:
        changed |= ensure_keyrings_dir(module)
        if not module.check_mode:
            module.atomic_move(tmpfile, filename)
        changed |= True
    changed |= module.set_mode_if_different(filename, 420, False)
    return (changed, filename, None)

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec={'allow_downgrade_to_insecure': {'type': 'bool'}, 'allow_insecure': {'type': 'bool'}, 'allow_weak': {'type': 'bool'}, 'architectures': {'elements': 'str', 'type': 'list'}, 'by_hash': {'type': 'bool'}, 'check_date': {'type': 'bool'}, 'check_valid_until': {'type': 'bool'}, 'components': {'elements': 'str', 'type': 'list'}, 'date_max_future': {'type': 'int'}, 'enabled': {'type': 'bool'}, 'inrelease_path': {'type': 'str'}, 'languages': {'elements': 'str', 'type': 'list'}, 'name': {'type': 'str', 'required': True}, 'pdiffs': {'type': 'bool'}, 'signed_by': {'type': 'str'}, 'suites': {'elements': 'str', 'type': 'list'}, 'targets': {'elements': 'str', 'type': 'list'}, 'trusted': {'type': 'bool'}, 'types': {'choices': ['deb', 'deb-src'], 'elements': 'str', 'type': 'list', 'default': ['deb']}, 'uris': {'elements': 'str', 'type': 'list'}, 'mode': {'type': 'raw', 'default': '0644'}, 'state': {'type': 'str', 'choices': ['present', 'absent'], 'default': 'present'}}, supports_check_mode=True)
    if not HAS_DEBIAN:
        module.fail_json(msg=missing_required_lib('python3-debian'), exception=DEBIAN_IMP_ERR)
    check_mode = module.check_mode
    changed = False
    params = module.params.copy()
    mode = params.pop('mode')
    state = params.pop('state')
    name = params['name']
    slug = re.sub('[^a-z0-9-]+', '', re.sub('[_\\s]+', '-', name.lower()))
    sources_filename = make_sources_filename(slug)
    if state == 'absent':
        if os.path.exists(sources_filename):
            if not check_mode:
                os.unlink(sources_filename)
            changed |= True
        for ext in ('asc', 'gpg'):
            signed_by_filename = make_signed_by_filename(slug, ext)
            if os.path.exists(signed_by_filename):
                if not check_mode:
                    os.unlink(signed_by_filename)
                changed = True
        module.exit_json(repo=None, changed=changed, dest=sources_filename, key_filename=signed_by_filename)
    deb822 = Deb822()
    signed_by_filename = None
    for (key, value) in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = format_bool(value)
        elif isinstance(value, int):
            value = to_native(value)
        elif is_sequence(value):
            value = format_list(value)
        elif key == 'signed_by':
            try:
                (key_changed, signed_by_filename, signed_by_data) = write_signed_by_key(module, value, slug)
                value = signed_by_filename or signed_by_data
                changed |= key_changed
            except RuntimeError as exc:
                module.fail_json(msg='Could not fetch signed_by key: %s' % to_native(exc))
        if value.count('\n') > 0:
            value = format_multiline(value)
        deb822[format_field_name(key)] = value
    repo = deb822.dump()
    (tmpfd, tmpfile) = tempfile.mkstemp(dir=module.tmpdir)
    with os.fdopen(tmpfd, 'wb') as f:
        f.write(to_bytes(repo))
    sources_filename = make_sources_filename(slug)
    src_chksum = module.sha256(tmpfile)
    dest_chksum = module.sha256(sources_filename)
    if src_chksum != dest_chksum:
        if not check_mode:
            module.atomic_move(tmpfile, sources_filename)
        changed |= True
    changed |= module.set_mode_if_different(sources_filename, mode, False)
    module.exit_json(repo=repo, changed=changed, dest=sources_filename, key_filename=signed_by_filename)
if __name__ == '__main__':
    main()