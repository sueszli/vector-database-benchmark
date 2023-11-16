import re
from flask_babel import gettext
from searx.botdetection._helpers import get_real_ip
name = gettext('Self Information')
description = gettext('Displays your IP if the query is "ip" and your user agent if the query contains "user agent".')
default_on = True
preference_section = 'query'
query_keywords = ['user-agent']
query_examples = ''
p = re.compile('.*user[ -]agent.*', re.IGNORECASE)

def post_search(request, search):
    if False:
        while True:
            i = 10
    if search.search_query.pageno > 1:
        return True
    if search.search_query.query == 'ip':
        ip = get_real_ip(request)
        search.result_container.answers['ip'] = {'answer': ip}
    elif p.match(search.search_query.query):
        ua = request.user_agent
        search.result_container.answers['user-agent'] = {'answer': ua}
    return True