"""Queryset for the redirects app."""
from urllib.parse import urlparse
import structlog
from django.db import models
from django.db.models import CharField, F, Q, Value
from readthedocs.core.permissions import AdminPermission
log = structlog.get_logger(__name__)

class RedirectQuerySet(models.QuerySet):
    """Redirects take into account their own privacy_level setting."""
    use_for_related_fields = True

    def _add_from_user_projects(self, queryset, user):
        if False:
            i = 10
            return i + 15
        if user.is_authenticated:
            projects_pk = AdminPermission.projects(user=user, admin=True, member=True).values_list('pk', flat=True)
            user_queryset = self.filter(project__in=projects_pk)
            queryset = user_queryset | queryset
        return queryset.distinct()

    def api(self, user=None):
        if False:
            i = 10
            return i + 15
        queryset = self.none()
        if user:
            queryset = self._add_from_user_projects(queryset, user)
        return queryset

    def get_redirect_path_with_status(self, path, full_path=None, language=None, version_slug=None, forced_only=False):
        if False:
            print('Hello World!')
        '\n        Get the final redirect with its status code.\n\n        :param path: Is the path without the language and version parts.\n        :param full_path: Is the full path including the language and version parts.\n        :param forced_only: Include only forced redirects in the results.\n        '
        if forced_only and (not self.filter(force=True).exists()):
            return (None, None)
        normalized_path = self._normalize_path(path)
        normalized_full_path = self._normalize_path(full_path)
        queryset = self.annotate(path=Value(normalized_path, output_field=CharField()), full_path=Value(normalized_full_path, output_field=CharField()))
        prefix = Q(redirect_type='prefix', full_path__startswith=F('from_url'))
        page = Q(redirect_type='page', path__exact=F('from_url'))
        exact = Q(redirect_type='exact', from_url__endswith='$rest', full_path__startswith=F('from_url_without_rest')) | Q(redirect_type='exact', full_path__exact=F('from_url'))
        sphinx_html = Q(redirect_type='sphinx_html', path__endswith='/') | Q(redirect_type='sphinx_html', path__endswith='/index.html')
        sphinx_htmldir = Q(redirect_type='sphinx_htmldir', path__endswith='.html')
        queryset = queryset.filter(prefix | page | exact | sphinx_html | sphinx_htmldir)
        if forced_only:
            queryset = queryset.filter(force=True)
        for redirect in queryset.select_related('project'):
            new_path = redirect.get_redirect_path(path=normalized_path, full_path=normalized_full_path, language=language, version_slug=version_slug)
            if new_path:
                return (new_path, redirect.http_status)
        return (None, None)

    def _normalize_path(self, path):
        if False:
            print('Hello World!')
        "\n        Normalize path.\n\n        We normalize ``path`` to:\n\n        - Remove the query params.\n        - Remove any invalid URL chars (\\r, \\n, \\t).\n        - Always start the path with ``/``.\n\n        We don't use ``.path`` to avoid parsing the filename as a full url.\n        For example if the path is ``http://example.com/my-path``,\n        ``.path`` would return ``my-path``.\n        "
        parsed_path = urlparse(path)
        normalized_path = parsed_path._replace(query='').geturl()
        normalized_path = '/' + normalized_path.lstrip('/')
        return normalized_path