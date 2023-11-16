from posthog.hogql_queries.query_runner import get_query_runner
from posthog.models import Team
from posthog.schema import DashboardFilter

def apply_dashboard_filters(query: dict, filters: dict, team: Team) -> dict:
    if False:
        for i in range(10):
            print('nop')
    kind = query.get('kind', None)
    if kind == 'DataTableNode':
        source = apply_dashboard_filters(query['source'], filters, team)
        return {**query, 'source': source}
    try:
        query_runner = get_query_runner(query, team)
    except ValueError:
        return query
    try:
        return query_runner.apply_dashboard_filters(DashboardFilter(**filters)).dict()
    except NotImplementedError:
        return query