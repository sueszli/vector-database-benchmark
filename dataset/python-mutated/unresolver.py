import re
from dataclasses import dataclass
from enum import Enum, auto
from urllib.parse import ParseResult, urlparse
import structlog
from django.conf import settings
from readthedocs.builds.constants import EXTERNAL, INTERNAL
from readthedocs.builds.models import Version
from readthedocs.constants import pattern_opts
from readthedocs.projects.models import Domain, Feature, Project
log = structlog.get_logger(__name__)

class UnresolverError(Exception):
    pass

class InvalidSchemeError(UnresolverError):

    def __init__(self, scheme):
        if False:
            i = 10
            return i + 15
        self.scheme = scheme

class InvalidXRTDSlugHeaderError(UnresolverError):
    pass

class DomainError(UnresolverError):

    def __init__(self, domain):
        if False:
            while True:
                i = 10
        self.domain = domain

class SuspiciousHostnameError(DomainError):
    pass

class InvalidSubdomainError(DomainError):
    pass

class InvalidExternalDomainError(DomainError):
    pass

class InvalidCustomDomainError(DomainError):
    pass

class VersionNotFoundError(UnresolverError):

    def __init__(self, project, version_slug, filename):
        if False:
            print('Hello World!')
        self.project = project
        self.version_slug = version_slug
        self.filename = filename

class TranslationNotFoundError(UnresolverError):

    def __init__(self, project, language, version_slug, filename):
        if False:
            print('Hello World!')
        self.project = project
        self.language = language
        self.filename = filename
        self.version_slug = version_slug

class TranslationWithoutVersionError(UnresolverError):

    def __init__(self, project, language):
        if False:
            i = 10
            return i + 15
        self.project = project
        self.language = language

class InvalidPathForVersionedProjectError(UnresolverError):

    def __init__(self, project, path):
        if False:
            i = 10
            return i + 15
        self.project = project
        self.path = path

class InvalidExternalVersionError(UnresolverError):

    def __init__(self, project, version_slug, external_version_slug):
        if False:
            i = 10
            return i + 15
        self.project = project
        self.version_slug = version_slug
        self.external_version_slug = external_version_slug

@dataclass(slots=True)
class UnresolvedURL:
    """Dataclass with the parts of an unresolved URL."""
    parent_project: Project
    project: Project
    version: Version
    filename: str
    parsed_url: ParseResult
    domain: Domain = None
    external: bool = False

class DomainSourceType(Enum):
    """Where the custom domain was resolved from."""
    custom_domain = auto()
    public_domain = auto()
    external_domain = auto()
    http_header = auto()

@dataclass(slots=True)
class UnresolvedDomain:
    source_domain: str
    source: DomainSourceType
    project: Project
    domain: Domain = None
    external_version_slug: str = None

    @property
    def is_from_custom_domain(self):
        if False:
            for i in range(10):
                print('nop')
        return self.source == DomainSourceType.custom_domain

    @property
    def is_from_public_domain(self):
        if False:
            print('Hello World!')
        return self.source == DomainSourceType.public_domain

    @property
    def is_from_http_header(self):
        if False:
            for i in range(10):
                print('nop')
        return self.source == DomainSourceType.http_header

    @property
    def is_from_external_domain(self):
        if False:
            while True:
                i = 10
        return self.source == DomainSourceType.external_domain

def _expand_regex(pattern):
    if False:
        return 10
    '\n    Expand a pattern with the patterns from pattern_opts.\n\n    This is used to avoid having a long regex.\n    '
    return re.compile(pattern.format(language=f"(?P<language>{pattern_opts['lang_slug']})", version=f"(?P<version>{pattern_opts['version_slug']})", filename=f"(?P<filename>{pattern_opts['filename_slug']})", subproject=f"(?P<subproject>{pattern_opts['project_slug']})"))

class Unresolver:
    multiversion_pattern = _expand_regex('^/{language}(/({version}(/{filename})?)?)?$')
    subproject_pattern = _expand_regex('^/{subproject}(/{filename})?$')

    def unresolve_url(self, url, append_indexhtml=True):
        if False:
            return 10
        '\n        Turn a URL into the component parts that our views would use to process them.\n\n        This is useful for lots of places,\n        like where we want to figure out exactly what file a URL maps to.\n\n        :param url: Full URL to unresolve (including the protocol and domain part).\n        :param append_indexhtml: If `True` directories will be normalized\n         to end with ``/index.html``.\n        '
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            raise InvalidSchemeError(parsed_url.scheme)
        domain = self.get_domain_from_host(parsed_url.netloc)
        unresolved_domain = self.unresolve_domain(domain)
        return self._unresolve(unresolved_domain=unresolved_domain, parsed_url=parsed_url, append_indexhtml=append_indexhtml)

    def unresolve_path(self, unresolved_domain, path, append_indexhtml=True):
        if False:
            print('Hello World!')
        "\n        Unresolve a path given a unresolved domain.\n\n        This is the same as the unresolve method,\n        but this method takes an unresolved domain\n        from unresolve_domain as input.\n\n        :param unresolved_domain: An UnresolvedDomain object.\n        :param path: Path to unresolve (this shouldn't include the protocol or querystrings).\n        :param append_indexhtml: If `True` directories will be normalized\n         to end with ``/index.html``.\n        "
        path = self._normalize_filename(path)
        parsed_url = ParseResult(scheme='', netloc='', path=path, params='', query='', fragment='')
        return self._unresolve(unresolved_domain=unresolved_domain, parsed_url=parsed_url, append_indexhtml=append_indexhtml)

    def _unresolve(self, unresolved_domain, parsed_url, append_indexhtml):
        if False:
            return 10
        '\n        The actual unresolver.\n\n        Extracted into a separate method so it can be re-used by\n        the unresolve and unresolve_path methods.\n        '
        (current_project, version, filename) = self._unresolve_path_with_parent_project(parent_project=unresolved_domain.project, path=parsed_url.path, external_version_slug=unresolved_domain.external_version_slug)
        if append_indexhtml and filename.endswith('/'):
            filename += 'index.html'
        return UnresolvedURL(parent_project=unresolved_domain.project, project=current_project, version=version, filename=filename, parsed_url=parsed_url, domain=unresolved_domain.domain, external=unresolved_domain.is_from_external_domain)

    @staticmethod
    def _normalize_filename(filename):
        if False:
            i = 10
            return i + 15
        'Normalize filename to always start with ``/``.'
        filename = filename or '/'
        if not filename.startswith('/'):
            filename = '/' + filename
        return filename

    def _match_multiversion_project(self, parent_project, path, external_version_slug=None):
        if False:
            return 10
        "\n        Try to match a multiversion project.\n\n        An exception is raised if we weren't able to find a matching version or language,\n        this exception has the current project (useful for 404 pages).\n\n        :returns: A tuple with the current project, version and filename.\n         Returns `None` if there isn't a total or partial match.\n        "
        custom_prefix = parent_project.custom_prefix
        if custom_prefix:
            if not path.startswith(custom_prefix):
                return None
            path = self._normalize_filename(path[len(custom_prefix):])
        match = self.multiversion_pattern.match(path)
        if not match:
            return None
        language = match.group('language')
        language = language.lower().replace('_', '-')
        version_slug = match.group('version')
        filename = self._normalize_filename(match.group('filename'))
        if parent_project.language == language:
            project = parent_project
        else:
            project = parent_project.translations.filter(language=language).first()
            if not project:
                raise TranslationNotFoundError(project=parent_project, language=language, version_slug=version_slug, filename=filename)
        if version_slug is None:
            raise TranslationWithoutVersionError(project=project, language=language)
        if external_version_slug and external_version_slug != version_slug:
            raise InvalidExternalVersionError(project=project, version_slug=version_slug, external_version_slug=external_version_slug)
        manager = EXTERNAL if external_version_slug else INTERNAL
        version = project.versions(manager=manager).filter(slug=version_slug).first()
        if not version:
            raise VersionNotFoundError(project=project, version_slug=version_slug, filename=filename)
        return (project, version, filename)

    def _match_subproject(self, parent_project, path, external_version_slug=None):
        if False:
            while True:
                i = 10
        "\n        Try to match a subproject.\n\n        If the subproject exists, we try to resolve the rest of the path\n        with the subproject as the canonical project.\n\n        :returns: A tuple with the current project, version and filename.\n         Returns `None` if there isn't a total or partial match.\n        "
        custom_prefix = parent_project.custom_subproject_prefix or '/projects/'
        if not path.startswith(custom_prefix):
            return None
        path = self._normalize_filename(path[len(custom_prefix):])
        match = self.subproject_pattern.match(path)
        if not match:
            return None
        subproject_alias = match.group('subproject')
        filename = self._normalize_filename(match.group('filename'))
        project_relationship = parent_project.subprojects.filter(alias=subproject_alias).select_related('child').first()
        if project_relationship:
            subproject = project_relationship.child
            response = self._unresolve_path_with_parent_project(parent_project=subproject, path=filename, check_subprojects=False, external_version_slug=external_version_slug)
            return response
        return None

    def _match_single_version_project(self, parent_project, path, external_version_slug=None):
        if False:
            while True:
                i = 10
        "\n        Try to match a single version project.\n\n        By default any path will match. If `external_version_slug` is given,\n        that version is used instead of the project's default version.\n\n        An exception is raised if we weren't able to find a matching version,\n        this exception has the current project (useful for 404 pages).\n\n        :returns: A tuple with the current project, version and filename.\n         Returns `None` if there isn't a total or partial match.\n        "
        custom_prefix = parent_project.custom_prefix
        if custom_prefix:
            if not path.startswith(custom_prefix):
                return None
            path = path[len(custom_prefix):]
        filename = self._normalize_filename(path)
        if external_version_slug:
            version_slug = external_version_slug
            manager = EXTERNAL
        else:
            version_slug = parent_project.default_version
            manager = INTERNAL
        version = parent_project.versions(manager=manager).filter(slug=version_slug).first()
        if not version:
            raise VersionNotFoundError(project=parent_project, version_slug=version_slug, filename=filename)
        return (parent_project, version, filename)

    def _unresolve_path_with_parent_project(self, parent_project, path, check_subprojects=True, external_version_slug=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Unresolve `path` with `parent_project` as base.\n\n        The returned project, version, and filename are guaranteed to not be\n        `None`. An exception is raised if we weren't able to resolve the\n        project, version or path/filename.\n\n        The checks are done in the following order:\n\n        - Check for multiple versions if the parent project\n          isn't a single version project.\n        - Check for subprojects.\n        - Check for single versions if the parent project isn't\n          a multi version project.\n\n        :param parent_project: The project that owns the path.\n        :param path: The path to unresolve.\n        :param check_subprojects: If we should check for subprojects,\n         this is used to call this function recursively when\n         resolving the path from a subproject (we don't support subprojects of subprojects).\n        :param external_version_slug: Slug of the external version.\n         Used instead of the default version for single version projects\n         being served under an external domain.\n\n        :returns: A tuple with: project, version, and filename.\n        "
        if not parent_project.single_version:
            response = self._match_multiversion_project(parent_project=parent_project, path=path, external_version_slug=external_version_slug)
            if response:
                return response
        if check_subprojects:
            response = self._match_subproject(parent_project=parent_project, path=path, external_version_slug=external_version_slug)
            if response:
                return response
        if parent_project.single_version:
            response = self._match_single_version_project(parent_project=parent_project, path=path, external_version_slug=external_version_slug)
            if response:
                return response
        raise InvalidPathForVersionedProjectError(project=parent_project, path=self._normalize_filename(path))

    @staticmethod
    def get_domain_from_host(host):
        if False:
            print('Hello World!')
        '\n        Get the normalized domain from a hostname.\n\n        A hostname can include the port.\n        '
        return host.lower().split(':')[0]

    def unresolve_domain(self, domain):
        if False:
            i = 10
            return i + 15
        '\n        Unresolve domain by extracting relevant information from it.\n\n        :param str domain: Domain to extract the information from.\n        :returns: A UnresolvedDomain object.\n        '
        public_domain = self.get_domain_from_host(settings.PUBLIC_DOMAIN)
        external_domain = self.get_domain_from_host(settings.RTD_EXTERNAL_VERSION_DOMAIN)
        (subdomain, *root_domain) = domain.split('.', maxsplit=1)
        root_domain = root_domain[0] if root_domain else ''
        if public_domain == root_domain:
            project_slug = subdomain
            log.debug('Public domain.', domain=domain)
            return UnresolvedDomain(source_domain=domain, source=DomainSourceType.public_domain, project=self._resolve_project_slug(project_slug, domain))
        if external_domain == root_domain:
            try:
                (project_slug, version_slug) = subdomain.rsplit('--', maxsplit=1)
                log.debug('External versions domain.', domain=domain)
                return UnresolvedDomain(source_domain=domain, source=DomainSourceType.external_domain, project=self._resolve_project_slug(project_slug, domain), external_version_slug=version_slug)
            except ValueError as exc:
                log.info('Invalid format of external versions domain.', domain=domain)
                raise InvalidExternalDomainError(domain=domain) from exc
        if public_domain in domain or external_domain in domain:
            log.warning('Weird variation of our domain.', domain=domain)
            raise SuspiciousHostnameError(domain=domain)
        domain_object = Domain.objects.filter(domain=domain).select_related('project').first()
        if not domain_object:
            log.info('Invalid domain.', domain=domain)
            raise InvalidCustomDomainError(domain=domain)
        log.debug('Custom domain.', domain=domain)
        return UnresolvedDomain(source_domain=domain, source=DomainSourceType.custom_domain, project=domain_object.project, domain=domain_object)

    def _resolve_project_slug(self, slug, domain):
        if False:
            return 10
        'Get the project from the slug or raise an exception if not found.'
        try:
            return Project.objects.get(slug=slug)
        except Project.DoesNotExist as exc:
            raise InvalidSubdomainError(domain=domain) from exc

    def unresolve_domain_from_request(self, request):
        if False:
            print('Hello World!')
        '\n        Unresolve domain by extracting relevant information from the request.\n\n        We first check if the ``X-RTD-Slug`` header has been set for explicit\n        project mapping, otherwise we unresolve by calling `self.unresolve_domain`\n        on the host.\n\n        :param request: Request to extract the information from.\n        :returns: A UnresolvedDomain object.\n        '
        host = self.get_domain_from_host(request.get_host())
        log.bind(host=host)
        header_project_slug = request.headers.get('X-RTD-Slug', '').lower()
        if header_project_slug:
            project = Project.objects.filter(slug=header_project_slug, feature__feature_id=Feature.RESOLVE_PROJECT_FROM_HEADER).first()
            if project:
                log.info('Setting project based on X_RTD_SLUG header.', project_slug=project.slug)
                return UnresolvedDomain(source_domain=host, source=DomainSourceType.http_header, project=project)
            log.warning('X-RTD-Header passed for project without it enabled.', project_slug=header_project_slug)
            raise InvalidXRTDSlugHeaderError
        return unresolver.unresolve_domain(host)
unresolver = Unresolver()
unresolve = unresolver.unresolve_url