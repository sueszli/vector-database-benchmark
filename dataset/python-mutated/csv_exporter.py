import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
import requests
import structlog
from django.http import QueryDict
from sentry_sdk import capture_exception, push_scope
from posthog.api.query import process_query
from posthog.jwt import PosthogJwtAudience, encode_jwt
from posthog.models.exported_asset import ExportedAsset, save_content
from posthog.utils import absolute_uri
from .ordered_csv_renderer import OrderedCsvRenderer
from ..exporter import EXPORT_FAILED_COUNTER, EXPORT_ASSET_UNKNOWN_COUNTER, EXPORT_SUCCEEDED_COUNTER, EXPORT_TIMER
from ...constants import CSV_EXPORT_LIMIT
logger = structlog.get_logger(__name__)

def add_query_params(url: str, params: Dict[str, str]) -> str:
    if False:
        while True:
            i = 10
    "\n    Uses parse_qsl because parse_qs turns all values into lists but doesn't unbox them when re-encoded\n    "
    parsed = urlparse(url)
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    update_params: List[Tuple[str, Any]] = []
    for (param, value) in query_params:
        if param in params:
            update_params.append((param, params.pop(param)))
        else:
            update_params.append((param, value))
    for (key, value) in params.items():
        update_params.append((key, value))
    encodedQueryParams = urlencode(update_params, quote_via=quote)
    parsed = parsed._replace(query=encodedQueryParams)
    return urlunparse(parsed)

def _convert_response_to_csv_data(data: Any) -> List[Any]:
    if False:
        i = 10
        return i + 15
    if isinstance(data.get('results'), list):
        results = data.get('results')
        if len(results) > 0 and (isinstance(results[0], list) or isinstance(results[0], tuple)) and ('types' in data):
            csv_rows: List[Dict[str, Any]] = []
            for row in results:
                row_dict = {}
                for (idx, x) in enumerate(row):
                    row_dict[data['columns'][idx]] = x
                csv_rows.append(row_dict)
            return csv_rows
        if len(results) == 1 and set(results[0].keys()) == {'people', 'count'}:
            return results[0].get('people')
        return results
    elif data.get('result') and isinstance(data.get('result'), list):
        items = data['result']
        first_result = items[0]
        if isinstance(first_result, list) or first_result.get('action_id'):
            csv_rows = []
            multiple_items = items if isinstance(first_result, list) else [items]
            for items in multiple_items:
                csv_rows.extend([{'name': x['custom_name'] or x['action_id'], 'breakdown_value': '::'.join(x.get('breakdown_value', [])), 'action_id': x['action_id'], 'count': x['count'], 'median_conversion_time (seconds)': x['median_conversion_time'], 'average_conversion_time (seconds)': x['average_conversion_time']} for x in items])
            return csv_rows
        elif first_result.get('appearances') and first_result.get('person'):
            csv_rows = []
            for item in items:
                line = {'person': item['person']['name']}
                for (index, data) in enumerate(item['appearances']):
                    line[f'Day {index}'] = data
                csv_rows.append(line)
            return csv_rows
        elif first_result.get('values') and first_result.get('label'):
            csv_rows = []
            for item in items:
                if item.get('date'):
                    line = {'cohort': item['date'], 'cohort size': item['values'][0]['count']}
                    for (index, data) in enumerate(item['values']):
                        line[items[index]['label']] = data['count']
                else:
                    line = {'cohort': item['label'], 'cohort size': item['values'][0]['count']}
                    for (index, data) in enumerate(item['values']):
                        line[f'Period {index}'] = data['count']
                csv_rows.append(line)
            return csv_rows
        elif isinstance(first_result.get('data'), list):
            csv_rows = []
            for (index, item) in enumerate(items):
                line = {'series': item.get('label', f'Series #{index + 1}')}
                if item.get('action', {}).get('custom_name'):
                    line['custom name'] = item.get('action').get('custom_name')
                if item.get('aggregated_value'):
                    line['total count'] = item.get('aggregated_value')
                else:
                    for (index, data) in enumerate(item['data']):
                        line[item['labels'][index]] = data
                csv_rows.append(line)
            return csv_rows
        else:
            return items
    return []

class UnexpectedEmptyJsonResponse(Exception):
    pass

def _export_to_csv(exported_asset: ExportedAsset, limit: int=1000) -> None:
    if False:
        return 10
    resource = exported_asset.export_context
    columns: List[str] = resource.get('columns', [])
    all_csv_rows: List[Any] = []
    if resource.get('source'):
        query = resource.get('source')
        query_response = process_query(team=exported_asset.team, query_json=query, in_export_context=True)
        all_csv_rows = _convert_response_to_csv_data(query_response)
    else:
        path: str = resource['path']
        method: str = resource.get('method', 'GET')
        body = resource.get('body', None)
        next_url = None
        access_token = encode_jwt({'id': exported_asset.created_by_id}, datetime.timedelta(minutes=15), PosthogJwtAudience.IMPERSONATED_USER)
        while len(all_csv_rows) < CSV_EXPORT_LIMIT:
            response = make_api_call(access_token, body, limit, method, next_url, path)
            if response.status_code != 200:
                try:
                    response_json = response.json()
                except Exception:
                    response_json = 'no response json to parse'
                raise Exception(f'export API call failed with status_code: {response.status_code}. {response_json}')
            data = response.json()
            if data is None:
                unexpected_empty_json_response = UnexpectedEmptyJsonResponse('JSON is None when calling API for data')
                logger.error('csv_exporter.json_was_none', exc=unexpected_empty_json_response, exc_info=True, response_text=response.text)
                raise unexpected_empty_json_response
            csv_rows = _convert_response_to_csv_data(data)
            all_csv_rows = all_csv_rows + csv_rows
            if not data.get('next') or not csv_rows:
                break
            next_url = data.get('next')
    renderer = OrderedCsvRenderer()
    if len(all_csv_rows):
        if not [x for x in all_csv_rows[0].values() if isinstance(x, dict) or isinstance(x, list)]:
            renderer.header = all_csv_rows[0].keys()
    render_context = {}
    if columns:
        render_context['header'] = columns
    rendered_csv_content = renderer.render(all_csv_rows, renderer_context=render_context)
    save_content(exported_asset, rendered_csv_content)

def get_limit_param_key(path: str) -> str:
    if False:
        while True:
            i = 10
    query = QueryDict(path)
    breakdown = query.get('breakdown', None)
    return 'breakdown_limit' if breakdown is not None else 'limit'

def make_api_call(access_token: str, body: Any, limit: int, method: str, next_url: Optional[str], path: str) -> requests.models.Response:
    if False:
        i = 10
        return i + 15
    request_url: str = absolute_uri(next_url or path)
    try:
        url = add_query_params(request_url, {get_limit_param_key(request_url): str(limit), 'is_csv_export': '1'})
        response = requests.request(method=method.lower(), url=url, json=body, headers={'Authorization': f'Bearer {access_token}'})
        return response
    except Exception as ex:
        logger.error('csv_exporter.error_making_api_call', exc=ex, exc_info=True, next_url=next_url, path=path, request_url=request_url, limit=limit)
        raise ex

def export_csv(exported_asset: ExportedAsset, limit: Optional[int]=None) -> None:
    if False:
        return 10
    if not limit:
        limit = 1000
    try:
        if exported_asset.export_format == 'text/csv':
            with EXPORT_TIMER.labels(type='csv').time():
                _export_to_csv(exported_asset, limit)
            EXPORT_SUCCEEDED_COUNTER.labels(type='csv').inc()
        else:
            EXPORT_ASSET_UNKNOWN_COUNTER.labels(type='csv').inc()
            raise NotImplementedError(f'Export to format {exported_asset.export_format} is not supported')
    except Exception as e:
        if exported_asset:
            team_id = str(exported_asset.team.id)
        else:
            team_id = 'unknown'
        with push_scope() as scope:
            scope.set_tag('celery_task', 'csv_export')
            scope.set_tag('team_id', team_id)
            capture_exception(e)
        logger.error('csv_exporter.failed', exception=e, exc_info=True)
        EXPORT_FAILED_COUNTER.labels(type='csv').inc()
        raise e