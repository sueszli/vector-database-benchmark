import email
import hashlib
import hmac
import os.path
import re
import tarfile
import tempfile
import zipfile
from cgi import FieldStorage
import packaging.requirements
import packaging.specifiers
import packaging.utils
import packaging.version
import packaging_legacy.version
import sentry_sdk
import wtforms
import wtforms.validators
from pyramid.httpexceptions import HTTPBadRequest, HTTPException, HTTPForbidden, HTTPGone, HTTPPermanentRedirect, HTTPTooManyRequests
from pyramid.response import Response
from pyramid.view import view_config
from sqlalchemy import func, orm
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from trove_classifiers import classifiers, deprecated_classifiers
from warehouse import forms
from warehouse.admin.flags import AdminFlagValue
from warehouse.classifiers.models import Classifier
from warehouse.email import send_basic_auth_with_two_factor_email, send_gpg_signature_uploaded_email, send_two_factor_not_yet_enabled_email
from warehouse.errors import BasicAuthTwoFactorEnabled
from warehouse.events.tags import EventTag
from warehouse.metrics import IMetricsService
from warehouse.packaging.interfaces import IFileStorage, IProjectService
from warehouse.packaging.models import Dependency, DependencyKind, Description, File, Filename, JournalEntry, Project, Release
from warehouse.packaging.tasks import sync_file_to_cache, update_bigquery_release_files
from warehouse.rate_limiting.interfaces import RateLimiterException
from warehouse.utils import http, readme
from warehouse.utils.project import PROJECT_NAME_RE, validate_project_name
from warehouse.utils.security_policy import AuthenticationMethod
ONE_MB = 1 * 1024 * 1024
ONE_GB = 1 * 1024 * 1024 * 1024
MAX_FILESIZE = 100 * ONE_MB
MAX_SIGSIZE = 8 * 1024
MAX_PROJECT_SIZE = 10 * ONE_GB
PATH_HASHER = 'blake2_256'
COMPRESSION_RATIO_MIN_SIZE = 64 * ONE_MB
COMPRESSION_RATIO_THRESHOLD = 50
_allowed_platforms = {'any', 'win32', 'win_arm64', 'win_amd64', 'win_ia64', 'manylinux1_x86_64', 'manylinux1_i686', 'manylinux2010_x86_64', 'manylinux2010_i686', 'manylinux2014_x86_64', 'manylinux2014_i686', 'manylinux2014_aarch64', 'manylinux2014_armv7l', 'manylinux2014_ppc64', 'manylinux2014_ppc64le', 'manylinux2014_s390x', 'linux_armv6l', 'linux_armv7l'}
_macosx_platform_re = re.compile('macosx_(?P<major>\\d+)_(\\d+)_(?P<arch>.*)')
_macosx_arches = {'ppc', 'ppc64', 'i386', 'x86_64', 'arm64', 'intel', 'fat', 'fat32', 'fat64', 'universal', 'universal2'}
_macosx_major_versions = {'10', '11', '12', '13', '14'}
_linux_platform_re = re.compile('(?P<libc>(many|musl))linux_(\\d+)_(\\d+)_(?P<arch>.*)')
_jointlinux_arches = {'x86_64', 'i686', 'aarch64', 'armv7l', 'ppc64le', 's390x'}
_manylinux_arches = _jointlinux_arches | {'ppc64'}
_musllinux_arches = _jointlinux_arches

def _valid_platform_tag(platform_tag):
    if False:
        while True:
            i = 10
    if platform_tag in _allowed_platforms:
        return True
    m = _macosx_platform_re.match(platform_tag)
    if m and m.group('major') in _macosx_major_versions and (m.group('arch') in _macosx_arches):
        return True
    m = _linux_platform_re.match(platform_tag)
    if m and m.group('libc') == 'musl':
        return m.group('arch') in _musllinux_arches
    if m and m.group('libc') == 'many':
        return m.group('arch') in _manylinux_arches
    return False
_error_message_order = ['metadata_version', 'name', 'version']
_dist_file_re = re.compile('.+?(?P<extension>\\.(tar\\.gz|zip|whl))$', re.I)
_legacy_specifier_re = re.compile('^(?P<name>\\S+)(?: \\((?P<specifier>\\S+)\\))?$')
_valid_description_content_types = {'text/plain', 'text/x-rst', 'text/markdown'}
_valid_markdown_variants = {'CommonMark', 'GFM'}
_filetype_extension_mapping = {'sdist': {'.zip', '.tar.gz'}, 'bdist_wheel': {'.whl'}}

def _exc_with_message(exc, message, **kwargs):
    if False:
        print('Hello World!')
    resp = exc(detail=message, **kwargs)
    status_message = message.encode('iso-8859-1', 'replace').decode('iso-8859-1')
    resp.status = f'{resp.status_code} {status_message}'
    return resp

def _validate_pep440_version(form, field):
    if False:
        i = 10
        return i + 15
    try:
        parsed = packaging.version.parse(field.data)
    except packaging.version.InvalidVersion:
        raise wtforms.validators.ValidationError("Start and end with a letter or numeral containing only ASCII numeric and '.', '_' and '-'.")
    if parsed.local is not None:
        raise wtforms.validators.ValidationError("Can't use PEP 440 local versions.")

def _parse_legacy_requirement(requirement):
    if False:
        for i in range(10):
            print('nop')
    parsed = _legacy_specifier_re.search(requirement)
    if parsed is None:
        raise ValueError('Invalid requirement.')
    return (parsed.groupdict()['name'], parsed.groupdict()['specifier'])

def _validate_pep440_specifier(specifier):
    if False:
        return 10
    try:
        packaging.specifiers.SpecifierSet(specifier)
    except packaging.specifiers.InvalidSpecifier:
        raise wtforms.validators.ValidationError('Invalid specifier in requirement.') from None

def _validate_pep440_specifier_field(form, field):
    if False:
        print('Hello World!')
    return _validate_pep440_specifier(field.data)

def _validate_legacy_non_dist_req(requirement):
    if False:
        i = 10
        return i + 15
    try:
        req = packaging.requirements.Requirement(requirement.replace('_', ''))
    except packaging.requirements.InvalidRequirement:
        raise wtforms.validators.ValidationError(f'Invalid requirement: {requirement!r}') from None
    if req.url is not None:
        raise wtforms.validators.ValidationError(f"Can't direct dependency: {requirement!r}")
    if any((not identifier.isalnum() or identifier[0].isdigit() for identifier in req.name.split('.'))):
        raise wtforms.validators.ValidationError('Use a valid Python identifier.')

def _validate_legacy_non_dist_req_list(form, field):
    if False:
        print('Hello World!')
    for datum in field.data:
        _validate_legacy_non_dist_req(datum)

def _validate_legacy_dist_req(requirement):
    if False:
        return 10
    try:
        req = packaging.requirements.Requirement(requirement)
    except packaging.requirements.InvalidRequirement:
        raise wtforms.validators.ValidationError(f'Invalid requirement: {requirement!r}.') from None
    if req.url is not None:
        raise wtforms.validators.ValidationError(f"Can't have direct dependency: {requirement!r}")

def _validate_legacy_dist_req_list(form, field):
    if False:
        return 10
    for datum in field.data:
        _validate_legacy_dist_req(datum)

def _validate_requires_external(requirement):
    if False:
        i = 10
        return i + 15
    (name, specifier) = _parse_legacy_requirement(requirement)
    if specifier is not None:
        _validate_pep440_specifier(specifier)

def _validate_requires_external_list(form, field):
    if False:
        for i in range(10):
            print('nop')
    for datum in field.data:
        _validate_requires_external(datum)

def _validate_project_url(value):
    if False:
        i = 10
        return i + 15
    try:
        (label, url) = (x.strip() for x in value.split(',', maxsplit=1))
    except ValueError:
        raise wtforms.validators.ValidationError('Use both a label and an URL.') from None
    if not label:
        raise wtforms.validators.ValidationError('Use a label.')
    if len(label) > 32:
        raise wtforms.validators.ValidationError('Use 32 characters or less.')
    if not url:
        raise wtforms.validators.ValidationError('Use an URL.')
    if not http.is_valid_uri(url, require_authority=False):
        raise wtforms.validators.ValidationError('Use valid URL.')

def _validate_project_url_list(form, field):
    if False:
        i = 10
        return i + 15
    for datum in field.data:
        _validate_project_url(datum)

def _validate_rfc822_email_field(form, field):
    if False:
        i = 10
        return i + 15
    email_validator = wtforms.validators.Email(message='Use a valid email address')
    addresses = email.utils.getaddresses([field.data])
    for (real_name, address) in addresses:
        email_validator(form, type('field', (), {'data': address}))

def _validate_description_content_type(form, field):
    if False:
        return 10

    def _raise(message):
        if False:
            for i in range(10):
                print('nop')
        raise wtforms.validators.ValidationError(f'Invalid description content type: {message}')
    msg = email.message.EmailMessage()
    msg['content-type'] = field.data
    (content_type, parameters) = (msg.get_content_type(), msg['content-type'].params)
    if content_type not in _valid_description_content_types:
        _raise('type/subtype is not valid')
    charset = parameters.get('charset')
    if charset and charset != 'UTF-8':
        _raise('Use a valid charset')
    variant = parameters.get('variant')
    if content_type == 'text/markdown' and variant and (variant not in _valid_markdown_variants):
        _raise('Use a valid variant, expected one of {}'.format(', '.join(_valid_markdown_variants)))

def _validate_no_deprecated_classifiers(form, field):
    if False:
        for i in range(10):
            print('nop')
    invalid_classifiers = set(field.data or []) & deprecated_classifiers.keys()
    if invalid_classifiers:
        first_invalid_classifier_name = sorted(invalid_classifiers)[0]
        deprecated_by = deprecated_classifiers[first_invalid_classifier_name]
        if deprecated_by:
            raise wtforms.validators.ValidationError(f'Classifier {first_invalid_classifier_name!r} has been deprecated, use the following classifier(s) instead: {deprecated_by}')
        else:
            raise wtforms.validators.ValidationError(f'Classifier {first_invalid_classifier_name!r} has been deprecated.')

def _validate_classifiers(form, field):
    if False:
        i = 10
        return i + 15
    invalid = sorted(set(field.data or []) - classifiers)
    if invalid:
        if len(invalid) == 1:
            raise wtforms.validators.ValidationError(f'Classifier {invalid[0]!r} is not a valid classifier.')
        else:
            raise wtforms.validators.ValidationError(f'Classifiers {invalid!r} are not valid classifiers.')

def _construct_dependencies(form, types):
    if False:
        return 10
    for (name, kind) in types.items():
        for item in getattr(form, name).data:
            yield Dependency(kind=kind.value, specifier=item)

class ListField(wtforms.Field):

    def process_formdata(self, valuelist):
        if False:
            for i in range(10):
                print('nop')
        self.data = [v.strip() for v in valuelist if v.strip()]

class MetadataForm(forms.Form):
    metadata_version = wtforms.StringField(description='Metadata-Version', validators=[wtforms.validators.InputRequired(), wtforms.validators.AnyOf(['1.0', '1.1', '1.2', '2.0', '2.1'], message='Use a known metadata version.')])
    name = wtforms.StringField(description='Name', validators=[wtforms.validators.InputRequired(), wtforms.validators.Regexp(PROJECT_NAME_RE, re.IGNORECASE, message="Start and end with a letter or numeral containing only ASCII numeric and '.', '_' and '-'.")])
    version = wtforms.StringField(description='Version', validators=[wtforms.validators.InputRequired(), wtforms.validators.Regexp('^(?!\\s).*(?<!\\s)$', message="Can't have leading or trailing whitespace."), _validate_pep440_version])
    summary = wtforms.StringField(description='Summary', validators=[wtforms.validators.Optional(), wtforms.validators.Length(max=512), wtforms.validators.Regexp('^.+$', message='Use a single line only.')])
    description = wtforms.StringField(description='Description', validators=[wtforms.validators.Optional()])
    author = wtforms.StringField(description='Author', validators=[wtforms.validators.Optional()])
    description_content_type = wtforms.StringField(description='Description-Content-Type', validators=[wtforms.validators.Optional(), _validate_description_content_type])
    author_email = wtforms.StringField(description='Author-email', validators=[wtforms.validators.Optional(), _validate_rfc822_email_field])
    maintainer = wtforms.StringField(description='Maintainer', validators=[wtforms.validators.Optional()])
    maintainer_email = wtforms.StringField(description='Maintainer-email', validators=[wtforms.validators.Optional(), _validate_rfc822_email_field])
    license = wtforms.StringField(description='License', validators=[wtforms.validators.Optional()])
    keywords = wtforms.StringField(description='Keywords', validators=[wtforms.validators.Optional()])
    classifiers = ListField(description='Classifier', validators=[_validate_no_deprecated_classifiers, _validate_classifiers])
    platform = wtforms.StringField(description='Platform', validators=[wtforms.validators.Optional()])
    home_page = wtforms.StringField(description='Home-Page', validators=[wtforms.validators.Optional(), forms.URIValidator()])
    download_url = wtforms.StringField(description='Download-URL', validators=[wtforms.validators.Optional(), forms.URIValidator()])
    requires_python = wtforms.StringField(description='Requires-Python', validators=[wtforms.validators.Optional(), _validate_pep440_specifier_field])
    pyversion = wtforms.StringField(validators=[wtforms.validators.Optional()])
    filetype = wtforms.StringField(validators=[wtforms.validators.InputRequired(), wtforms.validators.AnyOf(_filetype_extension_mapping.keys(), message='Use a known file type.')])
    comment = wtforms.StringField(validators=[wtforms.validators.Optional()])
    md5_digest = wtforms.StringField(validators=[wtforms.validators.Optional()])
    sha256_digest = wtforms.StringField(validators=[wtforms.validators.Optional(), wtforms.validators.Regexp('^[A-F0-9]{64}$', re.IGNORECASE, message='Use a valid, hex-encoded, SHA256 message digest.')])
    blake2_256_digest = wtforms.StringField(validators=[wtforms.validators.Optional(), wtforms.validators.Regexp('^[A-F0-9]{64}$', re.IGNORECASE, message='Use a valid, hex-encoded, BLAKE2 message digest.')])
    requires = ListField(validators=[wtforms.validators.Optional(), _validate_legacy_non_dist_req_list])
    provides = ListField(validators=[wtforms.validators.Optional(), _validate_legacy_non_dist_req_list])
    obsoletes = ListField(validators=[wtforms.validators.Optional(), _validate_legacy_non_dist_req_list])
    requires_dist = ListField(description='Requires-Dist', validators=[wtforms.validators.Optional(), _validate_legacy_dist_req_list])
    provides_dist = ListField(description='Provides-Dist', validators=[wtforms.validators.Optional(), _validate_legacy_dist_req_list])
    obsoletes_dist = ListField(description='Obsoletes-Dist', validators=[wtforms.validators.Optional(), _validate_legacy_dist_req_list])
    requires_external = ListField(description='Requires-External', validators=[wtforms.validators.Optional(), _validate_requires_external_list])
    project_urls = ListField(description='Project-URL', validators=[wtforms.validators.Optional(), _validate_project_url_list])

    def full_validate(self):
        if False:
            while True:
                i = 10
        if self.filetype.data and self.filetype.data != 'sdist' and (not self.pyversion.data):
            raise wtforms.validators.ValidationError('Python version is required for binary distribution uploads.')
        if self.filetype.data == 'sdist':
            if not self.pyversion.data:
                self.pyversion.data = 'source'
            elif self.pyversion.data != 'source':
                raise wtforms.validators.ValidationError("Use 'source' as Python version for an sdist.")
        if not self.md5_digest.data and (not self.sha256_digest.data) and (not self.blake2_256_digest.data):
            raise wtforms.validators.ValidationError('Include at least one message digest.')

def _validate_filename(filename, filetype):
    if False:
        print('Hello World!')
    disallowed = [*(chr(x) for x in range(32)), chr(127)]
    if [char for char in filename if char in disallowed]:
        raise _exc_with_message(HTTPBadRequest, 'Cannot upload a file with non-printable characters (ordinals 0-31) or the DEL character (ordinal 127) in the name.')
    if '/' in filename or '\\' in filename:
        raise _exc_with_message(HTTPBadRequest, "Cannot upload a file with '/' or '\\' in the name.")
    if (m := _dist_file_re.match(filename)):
        extension = m.group('extension')
        if extension not in _filetype_extension_mapping[filetype]:
            raise _exc_with_message(HTTPBadRequest, f'Invalid file extension: Extension {extension} is invalid for filetype {filetype}. See https://www.python.org/dev/peps/pep-0527 for more information.')
    else:
        raise _exc_with_message(HTTPBadRequest, 'Invalid file extension: Use .tar.gz, .whl or .zip extension. See https://www.python.org/dev/peps/pep-0527 and https://peps.python.org/pep-0715/ for more information')

def _is_valid_dist_file(filename, filetype):
    if False:
        i = 10
        return i + 15
    '\n    Perform some basic checks to see whether the indicated file could be\n    a valid distribution file.\n    '
    if zipfile.is_zipfile(filename):
        compressed_size = os.stat(filename).st_size
        with zipfile.ZipFile(filename) as zfp:
            decompressed_size = sum((e.file_size for e in zfp.infolist()))
        if decompressed_size > COMPRESSION_RATIO_MIN_SIZE and decompressed_size / compressed_size > COMPRESSION_RATIO_THRESHOLD:
            sentry_sdk.capture_message(f'File {filename} ({filetype}) exceeds compression ratio of {COMPRESSION_RATIO_THRESHOLD} ({decompressed_size}/{compressed_size})')
            return False
        with zipfile.ZipFile(filename) as zfp:
            for zinfo in zfp.infolist():
                if zinfo.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    return False
    if filename.endswith('.tar.gz'):
        try:
            with tarfile.open(filename, 'r:gz') as tar:
                bad_tar = True
                member = tar.next()
                while member:
                    parts = os.path.split(member.name)
                    if len(parts) == 2 and parts[1] == 'PKG-INFO':
                        bad_tar = False
                    member = tar.next()
                if bad_tar:
                    return False
        except (tarfile.ReadError, EOFError):
            return False
    elif filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(filename, 'r') as zfp:
                for zipname in zfp.namelist():
                    parts = os.path.split(zipname)
                    if len(parts) == 2 and parts[1] == 'PKG-INFO':
                        break
                else:
                    return False
        except zipfile.BadZipFile:
            return False
    elif filename.endswith('.whl'):
        try:
            with zipfile.ZipFile(filename, 'r') as zfp:
                for zipname in zfp.namelist():
                    parts = os.path.split(zipname)
                    if len(parts) == 2 and parts[1] == 'WHEEL':
                        break
                else:
                    return False
        except zipfile.BadZipFile:
            return False
    return True

def _is_duplicate_file(db_session, filename, hashes):
    if False:
        return 10
    "\n    Check to see if file already exists, and if it's content matches.\n    A file is considered to exist if its filename *or* blake2 digest are\n    present in a file row in the database.\n\n    Returns:\n    - True: This file is a duplicate and all further processing should halt.\n    - False: This file exists, but it is not a duplicate.\n    - None: This file does not exist.\n    "
    file_ = db_session.query(File).filter((File.filename == filename) | (File.blake2_256_digest == hashes['blake2_256'])).first()
    if file_ is not None:
        return file_.filename == filename and file_.sha256_digest == hashes['sha256'] and (file_.blake2_256_digest == hashes['blake2_256'])
    return None

@view_config(route_name='forklift.legacy.file_upload', uses_session=True, require_csrf=False, require_methods=['POST'], has_translations=True)
def file_upload(request):
    if False:
        while True:
            i = 10
    warnings = []
    if request.flags.enabled(AdminFlagValue.READ_ONLY):
        raise _exc_with_message(HTTPForbidden, 'Read-only mode: Uploads are temporarily disabled.')
    if request.flags.enabled(AdminFlagValue.DISALLOW_NEW_UPLOAD):
        raise _exc_with_message(HTTPForbidden, 'New uploads are temporarily disabled. See {projecthelp} for more information.'.format(projecthelp=request.help_url(_anchor='admin-intervention')))
    metrics = request.find_service(IMetricsService, context=None)
    metrics.increment('warehouse.upload.attempt')
    if request.identity is None:
        raise _exc_with_message(HTTPForbidden, 'Invalid or non-existent authentication information. See {projecthelp} for more information.'.format(projecthelp=request.help_url(_anchor='invalid-auth')))
    if request.user:
        if not (request.user.primary_email and request.user.primary_email.verified):
            raise _exc_with_message(HTTPBadRequest, 'User {!r} does not have a verified primary email address. Please add a verified primary email before attempting to upload to PyPI. See {project_help} for more information.'.format(request.user.username, project_help=request.help_url(_anchor='verified-email'))) from None
    for key in list(request.POST):
        value = request.POST.get(key)
        if isinstance(value, str):
            if value.strip() == 'UNKNOWN':
                del request.POST[key]
            if '\x00' in value:
                request.POST[key] = value.replace('\x00', '\\x00')
    if request.POST.get('protocol_version', '1') != '1':
        raise _exc_with_message(HTTPBadRequest, 'Unknown protocol version.')
    for field in set(request.POST) - {'content', 'gpg_signature'}:
        values = request.POST.getall(field)
        if any((isinstance(value, FieldStorage) for value in values)):
            raise _exc_with_message(HTTPBadRequest, f'{field}: Should not be a tuple.')
    form = MetadataForm(request.POST)
    if not form.validate():
        for field_name in _error_message_order:
            if field_name in form.errors:
                break
        else:
            field_name = sorted(form.errors.keys())[0]
        if field_name in form:
            field = form[field_name]
            if field.description and isinstance(field, wtforms.StringField):
                error_message = '{value!r} is an invalid value for {field}. '.format(value=field.data[:30] + '...' + field.data[-30:] if field.data and len(field.data) > 60 else field.data or '', field=field.description) + f'Error: {form.errors[field_name][0]} ' + 'See https://packaging.python.org/specifications/core-metadata' + ' for more information.'
            else:
                error_message = 'Invalid value for {field}. Error: {msgs[0]}'.format(field=field_name, msgs=form.errors[field_name])
        else:
            error_message = f'Error: {form.errors[field_name][0]}'
        raise _exc_with_message(HTTPBadRequest, error_message)
    if 'content' not in request.POST:
        raise _exc_with_message(HTTPBadRequest, 'Upload payload does not have a file.')
    project = request.db.query(Project).filter(Project.normalized_name == func.normalize_pep426_name(form.name.data)).first()
    if project is None:
        if not request.user:
            raise _exc_with_message(HTTPBadRequest, "Non-user identities cannot create new projects. This was probably caused by successfully using a pending publisher but specifying the project name incorrectly (either in the publisher or in your project's metadata). Please ensure that both match. See: https://docs.pypi.org/trusted-publishers/troubleshooting/")
        try:
            validate_project_name(form.name.data, request)
        except HTTPException as exc:
            raise _exc_with_message(exc.__class__, exc.detail) from None
        project_service = request.find_service(IProjectService)
        try:
            project = project_service.create_project(form.name.data, request.user, request)
        except RateLimiterException:
            msg = 'Too many new projects created'
            raise _exc_with_message(HTTPTooManyRequests, msg)
    allowed = request.has_permission('upload', project)
    if not allowed:
        reason = getattr(allowed, 'reason', None)
        if request.user:
            msg = "The user '{}' isn't allowed to upload to project '{}'. See {} for more information.".format(request.user.username, project.name, request.help_url(_anchor='project-name')) if reason is None else allowed.msg
        else:
            msg = "The given token isn't allowed to upload to project '{}'. See {} for more information.".format(project.name, request.help_url(_anchor='project-name')) if reason is None else allowed.msg
        raise _exc_with_message(HTTPForbidden, msg)
    if request.authentication_method == AuthenticationMethod.BASIC_AUTH and request.user.has_two_factor:
        send_basic_auth_with_two_factor_email(request, request.user, project_name=project.name)
        raise _exc_with_message(BasicAuthTwoFactorEnabled, f'User {request.user.username} has two factor auth enabled, an API Token or Trusted Publisher must be used to upload in place of password.')
    if project.name != form.name.data:
        project.name = form.name.data
    rendered = None
    if form.description.data:
        description_content_type = form.description_content_type.data
        if not description_content_type:
            description_content_type = 'text/x-rst'
        rendered = readme.render(form.description.data, description_content_type, use_fallback=False)
        if rendered is None:
            if form.description_content_type.data:
                message = "The description failed to render for '{description_content_type}'.".format(description_content_type=description_content_type)
            else:
                message = 'The description failed to render in the default format of reStructuredText.'
            raise _exc_with_message(HTTPBadRequest, '{message} See {projecthelp} for more information.'.format(message=message, projecthelp=request.help_url(_anchor='description-content-type'))) from None
    try:
        canonical_version = packaging.utils.canonicalize_version(form.version.data)
        release = request.db.query(Release).filter((Release.project == project) & (Release.canonical_version == canonical_version)).one()
    except MultipleResultsFound:
        release = request.db.query(Release).filter((Release.project == project) & (Release.version == form.version.data)).one()
    except NoResultFound:
        release_classifiers = request.db.query(Classifier).filter(Classifier.classifier.in_(form.classifiers.data)).all()
        project_urls = {name.strip(): url.strip() for (name, _, url) in (us.partition(',') for us in form.project_urls.data)}
        release = Release(project=project, _classifiers=release_classifiers, dependencies=list(_construct_dependencies(form, {'requires': DependencyKind.requires, 'provides': DependencyKind.provides, 'obsoletes': DependencyKind.obsoletes, 'requires_dist': DependencyKind.requires_dist, 'provides_dist': DependencyKind.provides_dist, 'obsoletes_dist': DependencyKind.obsoletes_dist, 'requires_external': DependencyKind.requires_external})), version=str(packaging.version.parse(form.version.data)), canonical_version=canonical_version, description=Description(content_type=form.description_content_type.data, raw=form.description.data or '', html=rendered or '', rendered_by=readme.renderer_version()), project_urls=project_urls, **{k: getattr(form, k).data for k in {'summary', 'license', 'author', 'author_email', 'maintainer', 'maintainer_email', 'keywords', 'platform', 'home_page', 'download_url', 'requires_python'}}, uploader=request.user if request.user else None, uploaded_via=request.user_agent)
        request.db.add(release)
        if 'gpg_signature' in request.POST:
            warnings.append('GPG signature support has been removed from PyPI and the provided signature has been discarded.')
            send_gpg_signature_uploaded_email(request, request.user, project_name=project.name)
        request.db.add(JournalEntry(name=release.project.name, version=release.version, action='new release', submitted_by=request.user if request.user else None))
        project.record_event(tag=EventTag.Project.ReleaseAdd, request=request, additional={'submitted_by': request.user.username if request.user else 'OpenID created token', 'canonical_version': release.canonical_version, 'publisher_url': request.oidc_publisher.publisher_url(request.oidc_claims) if request.oidc_publisher else None})
    releases = request.db.query(Release).filter(Release.project == project).options(orm.load_only(Release.project_id, Release.version, Release._pypi_ordering)).all()
    for (i, r) in enumerate(sorted(releases, key=lambda x: packaging_legacy.version.parse(x.version))):
        r._pypi_ordering = i
    filename = request.POST['content'].filename
    _validate_filename(filename, filetype=form.filetype.data)
    filename_prefix = filename.partition('-')[0] if filename.endswith('.whl') else filename.rpartition('-')[0]
    filename_prefix = filename_prefix.lower().replace('.', '_').replace('-', '_')
    if (prefix := project.normalized_name.replace('-', '_')) != filename_prefix:
        raise _exc_with_message(HTTPBadRequest, f'Start filename for {project.name!r} with {prefix!r}.')
    if not request.POST['content'].type or request.POST['content'].type.startswith('image/'):
        raise _exc_with_message(HTTPBadRequest, 'Invalid distribution file.')
    file_size_limit = max(filter(None, [MAX_FILESIZE, project.upload_limit]))
    project_size_limit = max(filter(None, [MAX_PROJECT_SIZE, project.total_size_limit]))
    file_data = None
    with tempfile.TemporaryDirectory() as tmpdir:
        temporary_filename = os.path.join(tmpdir, filename)
        with open(temporary_filename, 'wb') as fp:
            file_size = 0
            file_hashes = {'md5': hashlib.md5(usedforsecurity=False), 'sha256': hashlib.sha256(), 'blake2_256': hashlib.blake2b(digest_size=256 // 8)}
            metadata_file_hashes = {}
            for chunk in iter(lambda : request.POST['content'].file.read(8096), b''):
                file_size += len(chunk)
                if file_size > file_size_limit:
                    raise _exc_with_message(HTTPBadRequest, 'File too large. ' + 'Limit for project {name!r} is {limit} MB. '.format(name=project.name, limit=file_size_limit // ONE_MB) + 'See ' + request.help_url(_anchor='file-size-limit') + ' for more information.')
                if file_size + project.total_size > project_size_limit:
                    raise _exc_with_message(HTTPBadRequest, 'Project size too large. Limit for ' + 'project {name!r} total size is {limit} GB. '.format(name=project.name, limit=project_size_limit // ONE_GB) + 'See ' + request.help_url(_anchor='project-size-limit'))
                fp.write(chunk)
                for hasher in file_hashes.values():
                    hasher.update(chunk)
        file_hashes = {k: h.hexdigest().lower() for (k, h) in file_hashes.items()}
        if not all([hmac.compare_digest(getattr(form, f'{digest_name}_digest').data.lower(), digest_value) for (digest_name, digest_value) in file_hashes.items() if getattr(form, f'{digest_name}_digest').data]):
            raise _exc_with_message(HTTPBadRequest, 'The digest supplied does not match a digest calculated from the uploaded file.')
        is_duplicate = _is_duplicate_file(request.db, filename, file_hashes)
        if is_duplicate:
            request.tm.doom()
            return Response()
        elif is_duplicate is not None:
            raise _exc_with_message(HTTPBadRequest, 'File already exists. See ' + request.help_url(_anchor='file-name-reuse') + ' for more information.')
        if request.db.query(request.db.query(Filename).filter(Filename.filename == filename).exists()).scalar():
            raise _exc_with_message(HTTPBadRequest, 'This filename has already been used, use a different version. See ' + request.help_url(_anchor='file-name-reuse') + ' for more information.')
        if form.filetype.data == 'sdist' and request.db.query(request.db.query(File).filter((File.release == release) & (File.packagetype == 'sdist')).exists()).scalar():
            raise _exc_with_message(HTTPBadRequest, 'Only one sdist may be uploaded per release.')
        if not _is_valid_dist_file(temporary_filename, form.filetype.data):
            raise _exc_with_message(HTTPBadRequest, 'Invalid distribution file.')
        if filename.endswith('.whl'):
            try:
                (_, __, ___, tags) = packaging.utils.parse_wheel_filename(filename)
            except packaging.utils.InvalidWheelFilename as e:
                raise _exc_with_message(HTTPBadRequest, str(e))
            for tag in tags:
                if not _valid_platform_tag(tag.platform):
                    raise _exc_with_message(HTTPBadRequest, f"Binary wheel '{filename}' has an unsupported platform tag '{tag.platform}'.")
            '\n            Extract METADATA file from a wheel and return it as a content.\n            The name of the .whl file is used to find the corresponding .dist-info dir.\n            See https://peps.python.org/pep-0491/#file-contents\n            '
            filename = os.path.basename(temporary_filename)
            (name, version, _) = filename.split('-', 2)
            metadata_filename = f'{name}-{version}.dist-info/METADATA'
            try:
                with zipfile.ZipFile(temporary_filename) as zfp:
                    wheel_metadata_contents = zfp.read(metadata_filename)
            except KeyError:
                raise _exc_with_message(HTTPBadRequest, "Wheel '{filename}' does not contain the required METADATA file: {metadata_filename}".format(filename=filename, metadata_filename=metadata_filename))
            with open(temporary_filename + '.metadata', 'wb') as fp:
                fp.write(wheel_metadata_contents)
            metadata_file_hashes = {'sha256': hashlib.sha256(), 'blake2_256': hashlib.blake2b(digest_size=256 // 8)}
            for hasher in metadata_file_hashes.values():
                hasher.update(wheel_metadata_contents)
            metadata_file_hashes = {k: h.hexdigest().lower() for (k, h) in metadata_file_hashes.items()}
        request.db.add(Filename(filename=filename))
        file_ = File(release=release, filename=filename, python_version=form.pyversion.data, packagetype=form.filetype.data, comment_text=form.comment.data, size=file_size, md5_digest=file_hashes['md5'], sha256_digest=file_hashes['sha256'], blake2_256_digest=file_hashes['blake2_256'], metadata_file_sha256_digest=metadata_file_hashes.get('sha256'), metadata_file_blake2_256_digest=metadata_file_hashes.get('blake2_256'), path='/'.join([file_hashes[PATH_HASHER][:2], file_hashes[PATH_HASHER][2:4], file_hashes[PATH_HASHER][4:], filename]), uploaded_via=request.user_agent)
        file_data = file_
        request.db.add(file_)
        file_.record_event(tag=EventTag.File.FileAdd, request=request, additional={'filename': file_.filename, 'submitted_by': request.user.username if request.user else 'OpenID created token', 'canonical_version': release.canonical_version, 'publisher_url': request.oidc_publisher.publisher_url(request.oidc_claims) if request.oidc_publisher else None, 'project_id': str(project.id)})
        request.db.add(JournalEntry(name=release.project.name, version=release.version, action='add {python_version} file {filename}'.format(python_version=file_.python_version, filename=file_.filename), submitted_by=request.user if request.user else None))
        storage = request.find_service(IFileStorage, name='archive')
        storage.store(file_.path, os.path.join(tmpdir, filename), meta={'project': file_.release.project.normalized_name, 'version': file_.release.version, 'package-type': file_.packagetype, 'python-version': file_.python_version})
        if metadata_file_hashes:
            storage.store(file_.metadata_path, os.path.join(tmpdir, filename + '.metadata'), meta={'project': file_.release.project.normalized_name, 'version': file_.release.version, 'package-type': file_.packagetype, 'python-version': file_.python_version})
    if request.user and (not request.user.has_two_factor):
        warnings.append('Two factor authentication is not enabled for your account.')
        send_two_factor_not_yet_enabled_email(request, request.user)
    request.db.flush()
    dist_metadata = {'metadata_version': form['metadata_version'].data, 'name': form['name'].data, 'version': form['version'].data, 'summary': form['summary'].data, 'description': form['description'].data, 'author': form['author'].data, 'description_content_type': form['description_content_type'].data, 'author_email': form['author_email'].data, 'maintainer': form['maintainer'].data, 'maintainer_email': form['maintainer_email'].data, 'license': form['license'].data, 'keywords': form['keywords'].data, 'classifiers': form['classifiers'].data, 'platform': form['platform'].data, 'home_page': form['home_page'].data, 'download_url': form['download_url'].data, 'requires_python': form['requires_python'].data, 'pyversion': form['pyversion'].data, 'filetype': form['filetype'].data, 'comment': form['comment'].data, 'requires': form['requires'].data, 'provides': form['provides'].data, 'obsoletes': form['obsoletes'].data, 'requires_dist': form['requires_dist'].data, 'provides_dist': form['provides_dist'].data, 'obsoletes_dist': form['obsoletes_dist'].data, 'requires_external': form['requires_external'].data, 'project_urls': form['project_urls'].data, 'filename': file_data.filename, 'python_version': file_data.python_version, 'packagetype': file_data.packagetype, 'comment_text': file_data.comment_text, 'size': file_data.size, 'has_signature': False, 'md5_digest': file_data.md5_digest, 'sha256_digest': file_data.sha256_digest, 'blake2_256_digest': file_data.blake2_256_digest, 'path': file_data.path, 'uploaded_via': file_data.uploaded_via, 'upload_time': file_data.upload_time}
    if not request.registry.settings.get('warehouse.release_files_table') is None:
        request.task(update_bigquery_release_files).delay(dist_metadata)
    metrics.increment('warehouse.upload.ok', tags=[f'filetype:{form.filetype.data}'])
    request.task(sync_file_to_cache).delay(file_.id)
    return Response('\n'.join(warnings))

@view_config(route_name='forklift.legacy.submit', require_csrf=False, require_methods=['POST'])
@view_config(route_name='forklift.legacy.submit_pkg_info', require_csrf=False, require_methods=['POST'])
def submit(request):
    if False:
        i = 10
        return i + 15
    return _exc_with_message(HTTPGone, 'Project pre-registration is no longer required or supported, upload your files instead.')

@view_config(route_name='forklift.legacy.doc_upload', require_csrf=False, require_methods=['POST'])
def doc_upload(request):
    if False:
        i = 10
        return i + 15
    return _exc_with_message(HTTPGone, 'Uploading documentation is no longer supported, we recommend using https://readthedocs.org/.')

@view_config(route_name='forklift.legacy.missing_trailing_slash', require_csrf=False, require_methods=['POST'])
def missing_trailing_slash_redirect(request):
    if False:
        while True:
            i = 10
    '\n    Redirect requests to /legacy to the correct /legacy/ route with a\n    HTTP-308 Permanent Redirect\n    '
    return _exc_with_message(HTTPPermanentRedirect, 'An upload was attempted to /legacy but the expected upload URL is /legacy/ (with a trailing slash)', location=request.route_path('forklift.legacy.file_upload'))