import datetime
from django.conf import settings
from django.views import View
from django.shortcuts import render

class AppServers(View):

    def get(self, request):
        if False:
            while True:
                i = 10
        data = dict((('%s' % s['_id'].replace('-', ''), s['feeds']) for s in self.stats))
        chart_name = 'app_servers'
        chart_type = 'counter'
        formatted_data = {}
        for (k, v) in data.items():
            formatted_data[k] = f'{chart_name}{{app_server="{k}"}} {v}'
        context = {'data': formatted_data, 'chart_name': chart_name, 'chart_type': chart_type}
        return render(request, 'monitor/prometheus_data.html', context, content_type='text/plain')

    @property
    def stats(self):
        if False:
            i = 10
            return i + 15
        stats = settings.MONGOANALYTICSDB.nbanalytics.page_loads.aggregate([{'$match': {'date': {'$gte': datetime.datetime.now() - datetime.timedelta(minutes=5)}}}, {'$group': {'_id': '$server', 'feeds': {'$sum': 1}}}])
        return list(stats)

    @property
    def total(self):
        if False:
            for i in range(10):
                print('nop')
        stats = settings.MONGOANALYTICSDB.nbanalytics.page_loads.aggregate([{'$match': {'date': {'$gt': datetime.datetime.now() - datetime.timedelta(minutes=5)}}}, {'$group': {'_id': 1, 'feeds': {'$sum': 1}}}])
        return list(stats)