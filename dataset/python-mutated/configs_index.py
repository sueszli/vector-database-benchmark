from django.http.response import Http404
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.api.serializers import serialize
from sentry.api.serializers.models.plugin import PluginSerializer
from sentry.constants import ObjectStatus
from sentry.models.options.project_option import ProjectOption
from sentry.models.project import Project
from sentry.plugins.base import plugins

@region_silo_endpoint
class OrganizationPluginsConfigsEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        List one or more plugin configurations, including a `projectList` for each plugin which contains\n        all the projects that have that specific plugin both configured and enabled.\n\n        - similar to the `OrganizationPluginsEndpoint`, and can eventually replace it\n\n        :qparam plugins array[string]: an optional list of plugin ids (slugs) if you want specific plugins.\n                                    If not set, will return configurations for all plugins.\n        '
        desired_plugins = []
        for slug in request.GET.getlist('plugins') or ():
            try:
                desired_plugins.append(plugins.get(slug))
            except KeyError:
                return Response({'detail': 'Plugin %s not found' % slug}, status=404)
        if not desired_plugins:
            desired_plugins = list(plugins.plugin_that_can_be_configured())
        keys_to_check = []
        for plugin in desired_plugins:
            keys_to_check.append('%s:enabled' % plugin.slug)
            if plugin.required_field:
                keys_to_check.append(f'{plugin.slug}:{plugin.required_field}')
        project_options = ProjectOption.objects.filter(key__in=keys_to_check, project__organization=organization).exclude(value__in=[False, ''])
        '\n        This map stores info about whether a plugin is configured and/or enabled\n        {\n            "plugin_slug": {\n                "project_id": { "enabled": True, "configured": False },\n            },\n        }\n        '
        info_by_plugin_project = {}
        for project_option in project_options:
            [slug, field] = project_option.key.split(':')
            project_id = project_option.project_id
            info_by_plugin_project.setdefault(slug, {}).setdefault(project_id, {'enabled': False, 'configured': False})
            if field == 'enabled':
                info_by_plugin_project[slug][project_id]['enabled'] = True
            else:
                info_by_plugin_project[slug][project_id]['configured'] = True
        project_id_set = {project_option.project_id for project_option in project_options}
        projects = Project.objects.filter(id__in=project_id_set, status=ObjectStatus.ACTIVE)
        project_map = {project.id: project for project in projects}
        serialized_plugins = []
        for plugin in desired_plugins:
            serialized_plugin = serialize(plugin, request.user, PluginSerializer())
            if serialized_plugin['isDeprecated']:
                continue
            serialized_plugin['projectList'] = []
            info_by_project = info_by_plugin_project.get(plugin.slug, {})
            for (project_id, plugin_info) in info_by_project.items():
                if project_id not in project_map:
                    continue
                project = project_map[project_id]
                if not plugin_info['configured']:
                    continue
                serialized_plugin['projectList'].append({'projectId': project.id, 'projectSlug': project.slug, 'projectName': project.name, 'enabled': plugin_info['enabled'], 'configured': plugin_info['configured'], 'projectPlatform': project.platform})
            serialized_plugin['projectList'].sort(key=lambda x: x['projectSlug'])
            serialized_plugins.append(serialized_plugin)
        if not serialized_plugins:
            raise Http404
        return Response(serialized_plugins)