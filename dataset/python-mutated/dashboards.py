from flask import request, url_for
from flask_restful import abort
from funcy import partial, project
from sqlalchemy.orm.exc import StaleDataError
from redash import models
from redash.handlers.base import BaseResource, filter_by_tags, get_object_or_404, paginate
from redash.handlers.base import order_results as _order_results
from redash.permissions import can_modify, require_admin_or_owner, require_object_modify_permission, require_permission
from redash.security import csp_allows_embeding
from redash.serializers import DashboardSerializer, public_dashboard
order_map = {'name': 'lowercase_name', '-name': '-lowercase_name', 'created_at': 'created_at', '-created_at': '-created_at'}
order_results = partial(_order_results, default_order='-created_at', allowed_orders=order_map)

class DashboardListResource(BaseResource):

    @require_permission('list_dashboards')
    def get(self):
        if False:
            print('Hello World!')
        '\n        Lists all accessible dashboards.\n\n        :qparam number page_size: Number of queries to return per page\n        :qparam number page: Page number to retrieve\n        :qparam number order: Name of column to order by\n        :qparam number q: Full text search term\n\n        Responds with an array of :ref:`dashboard <dashboard-response-label>`\n        objects.\n        '
        search_term = request.args.get('q')
        if search_term:
            results = models.Dashboard.search(self.current_org, self.current_user.group_ids, self.current_user.id, search_term)
        else:
            results = models.Dashboard.all(self.current_org, self.current_user.group_ids, self.current_user.id)
        results = filter_by_tags(results, models.Dashboard.tags)
        ordered_results = order_results(results, fallback=not bool(search_term))
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 25, type=int)
        response = paginate(ordered_results, page=page, page_size=page_size, serializer=DashboardSerializer)
        if search_term:
            self.record_event({'action': 'search', 'object_type': 'dashboard', 'term': search_term})
        else:
            self.record_event({'action': 'list', 'object_type': 'dashboard'})
        return response

    @require_permission('create_dashboard')
    def post(self):
        if False:
            while True:
                i = 10
        '\n        Creates a new dashboard.\n\n        :<json string name: Dashboard name\n\n        Responds with a :ref:`dashboard <dashboard-response-label>`.\n        '
        dashboard_properties = request.get_json(force=True)
        dashboard = models.Dashboard(name=dashboard_properties['name'], org=self.current_org, user=self.current_user, is_draft=True, layout='[]')
        models.db.session.add(dashboard)
        models.db.session.commit()
        return DashboardSerializer(dashboard).serialize()

class MyDashboardsResource(BaseResource):

    @require_permission('list_dashboards')
    def get(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve a list of dashboards created by the current user.\n\n        :qparam number page_size: Number of dashboards to return per page\n        :qparam number page: Page number to retrieve\n        :qparam number order: Name of column to order by\n        :qparam number search: Full text search term\n\n        Responds with an array of :ref:`dashboard <dashboard-response-label>`\n        objects.\n        '
        search_term = request.args.get('q', '')
        if search_term:
            results = models.Dashboard.search_by_user(search_term, self.current_user)
        else:
            results = models.Dashboard.by_user(self.current_user)
        results = filter_by_tags(results, models.Dashboard.tags)
        ordered_results = order_results(results, fallback=not bool(search_term))
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 25, type=int)
        return paginate(ordered_results, page, page_size, DashboardSerializer)

class DashboardResource(BaseResource):

    @require_permission('list_dashboards')
    def get(self, dashboard_id=None):
        if False:
            print('Hello World!')
        '\n        Retrieves a dashboard.\n\n        :qparam number id: Id of dashboard to retrieve.\n\n        .. _dashboard-response-label:\n\n        :>json number id: Dashboard ID\n        :>json string name:\n        :>json string slug:\n        :>json number user_id: ID of the dashboard creator\n        :>json string created_at: ISO format timestamp for dashboard creation\n        :>json string updated_at: ISO format timestamp for last dashboard modification\n        :>json number version: Revision number of dashboard\n        :>json boolean dashboard_filters_enabled: Whether filters are enabled or not\n        :>json boolean is_archived: Whether this dashboard has been removed from the index or not\n        :>json boolean is_draft: Whether this dashboard is a draft or not.\n        :>json array layout: Array of arrays containing widget IDs, corresponding to the rows and columns the widgets are displayed in\n        :>json array widgets: Array of arrays containing :ref:`widget <widget-response-label>` data\n        :>json object options: Dashboard options\n\n        .. _widget-response-label:\n\n        Widget structure:\n\n        :>json number widget.id: Widget ID\n        :>json number widget.width: Widget size\n        :>json object widget.options: Widget options\n        :>json number widget.dashboard_id: ID of dashboard containing this widget\n        :>json string widget.text: Widget contents, if this is a text-box widget\n        :>json object widget.visualization: Widget contents, if this is a visualization widget\n        :>json string widget.created_at: ISO format timestamp for widget creation\n        :>json string widget.updated_at: ISO format timestamp for last widget modification\n        '
        if request.args.get('legacy') is not None:
            fn = models.Dashboard.get_by_slug_and_org
        else:
            fn = models.Dashboard.get_by_id_and_org
        dashboard = get_object_or_404(fn, dashboard_id, self.current_org)
        response = DashboardSerializer(dashboard, with_widgets=True, user=self.current_user).serialize()
        api_key = models.ApiKey.get_by_object(dashboard)
        if api_key:
            response['public_url'] = url_for('redash.public_dashboard', token=api_key.api_key, org_slug=self.current_org.slug, _external=True)
            response['api_key'] = api_key.api_key
        response['can_edit'] = can_modify(dashboard, self.current_user)
        self.record_event({'action': 'view', 'object_id': dashboard.id, 'object_type': 'dashboard'})
        return response

    @require_permission('edit_dashboard')
    def post(self, dashboard_id):
        if False:
            return 10
        '\n        Modifies a dashboard.\n\n        :qparam number id: Id of dashboard to retrieve.\n\n        Responds with the updated :ref:`dashboard <dashboard-response-label>`.\n\n        :status 200: success\n        :status 409: Version conflict -- dashboard modified since last read\n        '
        dashboard_properties = request.get_json(force=True)
        dashboard = models.Dashboard.get_by_id_and_org(dashboard_id, self.current_org)
        require_object_modify_permission(dashboard, self.current_user)
        updates = project(dashboard_properties, ('name', 'layout', 'version', 'tags', 'is_draft', 'is_archived', 'dashboard_filters_enabled', 'options'))
        if 'version' in updates and updates['version'] != dashboard.version:
            abort(409)
        updates['changed_by'] = self.current_user
        self.update_model(dashboard, updates)
        models.db.session.add(dashboard)
        try:
            models.db.session.commit()
        except StaleDataError:
            abort(409)
        result = DashboardSerializer(dashboard, with_widgets=True, user=self.current_user).serialize()
        self.record_event({'action': 'edit', 'object_id': dashboard.id, 'object_type': 'dashboard'})
        return result

    @require_permission('edit_dashboard')
    def delete(self, dashboard_id):
        if False:
            return 10
        '\n        Archives a dashboard.\n\n        :qparam number id: Id of dashboard to retrieve.\n\n        Responds with the archived :ref:`dashboard <dashboard-response-label>`.\n        '
        dashboard = models.Dashboard.get_by_id_and_org(dashboard_id, self.current_org)
        dashboard.is_archived = True
        dashboard.record_changes(changed_by=self.current_user)
        models.db.session.add(dashboard)
        d = DashboardSerializer(dashboard, with_widgets=True, user=self.current_user).serialize()
        models.db.session.commit()
        self.record_event({'action': 'archive', 'object_id': dashboard.id, 'object_type': 'dashboard'})
        return d

class PublicDashboardResource(BaseResource):
    decorators = BaseResource.decorators + [csp_allows_embeding]

    def get(self, token):
        if False:
            print('Hello World!')
        '\n        Retrieve a public dashboard.\n\n        :param token: An API key for a public dashboard.\n        :>json array widgets: An array of arrays of :ref:`public widgets <public-widget-label>`, corresponding to the rows and columns the widgets are displayed in\n        '
        if self.current_org.get_setting('disable_public_urls'):
            abort(400, message='Public URLs are disabled.')
        if not isinstance(self.current_user, models.ApiUser):
            api_key = get_object_or_404(models.ApiKey.get_by_api_key, token)
            dashboard = api_key.object
        else:
            dashboard = self.current_user.object
        return public_dashboard(dashboard)

class DashboardShareResource(BaseResource):

    def post(self, dashboard_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Allow anonymous access to a dashboard.\n\n        :param dashboard_id: The numeric ID of the dashboard to share.\n        :>json string public_url: The URL for anonymous access to the dashboard.\n        :>json api_key: The API key to use when accessing it.\n        '
        dashboard = models.Dashboard.get_by_id_and_org(dashboard_id, self.current_org)
        require_admin_or_owner(dashboard.user_id)
        api_key = models.ApiKey.create_for_object(dashboard, self.current_user)
        models.db.session.flush()
        models.db.session.commit()
        public_url = url_for('redash.public_dashboard', token=api_key.api_key, org_slug=self.current_org.slug, _external=True)
        self.record_event({'action': 'activate_api_key', 'object_id': dashboard.id, 'object_type': 'dashboard'})
        return {'public_url': public_url, 'api_key': api_key.api_key}

    def delete(self, dashboard_id):
        if False:
            while True:
                i = 10
        '\n        Disable anonymous access to a dashboard.\n\n        :param dashboard_id: The numeric ID of the dashboard to unshare.\n        '
        dashboard = models.Dashboard.get_by_id_and_org(dashboard_id, self.current_org)
        require_admin_or_owner(dashboard.user_id)
        api_key = models.ApiKey.get_by_object(dashboard)
        if api_key:
            api_key.active = False
            models.db.session.add(api_key)
            models.db.session.commit()
        self.record_event({'action': 'deactivate_api_key', 'object_id': dashboard.id, 'object_type': 'dashboard'})

class DashboardTagsResource(BaseResource):

    @require_permission('list_dashboards')
    def get(self):
        if False:
            while True:
                i = 10
        '\n        Lists all accessible dashboards.\n        '
        tags = models.Dashboard.all_tags(self.current_org, self.current_user)
        return {'tags': [{'name': name, 'count': count} for (name, count) in tags]}

class DashboardFavoriteListResource(BaseResource):

    def get(self):
        if False:
            print('Hello World!')
        search_term = request.args.get('q')
        if search_term:
            base_query = models.Dashboard.search(self.current_org, self.current_user.group_ids, self.current_user.id, search_term)
            favorites = models.Dashboard.favorites(self.current_user, base_query=base_query)
        else:
            favorites = models.Dashboard.favorites(self.current_user)
        favorites = filter_by_tags(favorites, models.Dashboard.tags)
        favorites = order_results(favorites, fallback=not bool(search_term))
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 25, type=int)
        response = paginate(favorites, page, page_size, DashboardSerializer)
        self.record_event({'action': 'load_favorites', 'object_type': 'dashboard', 'params': {'q': search_term, 'tags': request.args.getlist('tags'), 'page': page}})
        return response

class DashboardForkResource(BaseResource):

    @require_permission('edit_dashboard')
    def post(self, dashboard_id):
        if False:
            i = 10
            return i + 15
        dashboard = models.Dashboard.get_by_id_and_org(dashboard_id, self.current_org)
        fork_dashboard = dashboard.fork(self.current_user)
        models.db.session.commit()
        self.record_event({'action': 'fork', 'object_id': dashboard_id, 'object_type': 'dashboard'})
        return DashboardSerializer(fork_dashboard, with_widgets=True).serialize()