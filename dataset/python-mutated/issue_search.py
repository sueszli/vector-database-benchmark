import re
from django.db.models import Q
from plane.db.models import Issue

def search_issues(query, queryset):
    if False:
        while True:
            i = 10
    fields = ['name', 'sequence_id']
    q = Q()
    for field in fields:
        if field == 'sequence_id':
            sequences = re.findall('\\d+\\.\\d+|\\d+', query)
            for sequence_id in sequences:
                q |= Q(**{'sequence_id': sequence_id})
        else:
            q |= Q(**{f'{field}__icontains': query})
    return queryset.filter(q).distinct()