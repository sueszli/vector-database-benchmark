import json
import requests
from pylons import app_globals as g

def post_takedown_notice_to_external_site(title, request_type, date_sent, date_received, source, action_taken, public_description, kind, original_url, infringing_urls, submitter_attributes, sender_name, sender_kind, sender_country):
    if False:
        return 10
    'This method publicly posts a copy of the takedown notice to \n    https://lumendatabase.org. Posting notices to Lumen is free, and needs to\n    be arranged by contacting their team. Read more about Lumen at\n    https://www.lumendatabase.org/pages/about\n    '
    notice_json = {'authentication_token': g.secrets['lumendatabase_org_api_key'], 'notice': {'title': title, 'type': request_type, 'date_sent': date_sent.strftime('%Y-%m-%d'), 'date_received': date_received.strftime('%Y-%m-%d'), 'source': source, 'jurisdiction_list': 'US, CA', 'action_taken': action_taken, 'works_attributes': [{'description': public_description, 'kind': kind, 'copyrighted_urls_attributes': [{'url': original_url}], 'infringing_urls_attributes': infringing_urls}], 'entity_notice_roles_attributes': [{'name': 'recipient', 'entity_attributes': submitter_attributes}, {'name': 'sender', 'entity_attributes': {'name': sender_name, 'kind': sender_kind, 'address_line_1': '', 'city': '', 'state': '', 'zip': '', 'country_code': sender_country}}]}}
    timer = g.stats.get_timer('lumendatabase.takedown_create')
    timer.start()
    response = requests.post('%snotices' % g.live_config['lumendatabase_org_api_base_url'], headers={'Content-type': 'application/json', 'Accept': 'application/json'}, data=json.dumps(notice_json))
    timer.stop()
    return response.headers['location']