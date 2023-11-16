"""Template tags to query projects by privacy."""
from django import template
from django.db.models import Exists, OuterRef
from readthedocs.builds.models import Build
from readthedocs.core.permissions import AdminPermission
from readthedocs.projects.models import Project
register = template.Library()

@register.filter
def is_admin(user, project):
    if False:
        while True:
            i = 10
    return AdminPermission.is_admin(user, project)

@register.filter
def is_member(user, project):
    if False:
        for i in range(10):
            print('nop')
    return AdminPermission.is_member(user, project)

@register.simple_tag(takes_context=True)
def get_public_projects(context, user):
    if False:
        while True:
            i = 10
    projects = Project.objects.for_user_and_viewer(user=user, viewer=context['request'].user).prefetch_latest_build().annotate(_good_build=Exists(Build.internal.filter(success=True, project=OuterRef('pk'))))
    context['public_projects'] = projects
    return ''