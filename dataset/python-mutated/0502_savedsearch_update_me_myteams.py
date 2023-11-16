import re
from typing import List, Optional, Set, cast
from django.db import migrations
from sentry.api.issue_search import parse_search_query
from sentry.exceptions import InvalidSearchQuery
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def replacement_term(original_term: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    try:
        search_filters = parse_search_query(original_term)
    except Exception:
        return None
    if len(search_filters) != 1:
        raise Exception(f"'{original_term}' should only include a single parselable SearchFilter but {len(search_filters)} were parsed")
    assigned_filter = search_filters[0]
    in_syntax = isinstance(assigned_filter.value.raw_value, list)
    vals_set: Set[str] = set(cast(list, assigned_filter.value.raw_value)) if in_syntax else {cast(str, assigned_filter.value.raw_value)}
    if 'my_teams' in vals_set:
        return None
    elif 'me' in vals_set:
        search_filter_values: List[str] = list(cast(list, assigned_filter.value.raw_value)) if in_syntax else [cast(str, assigned_filter.value.raw_value)]
        if None in vals_set:
            search_filter_values = [v if v is not None else 'none' for v in search_filter_values]
        for (i, v) in enumerate(search_filter_values):
            if v == 'me':
                search_filter_values.insert(i + 1, 'my_teams')
                break
        for (i, v) in enumerate(search_filter_values):
            if ' ' in v:
                search_filter_values[i] = f'"{v}"'
        joined = ', '.join(search_filter_values)
        search_filter_key = 'assigned' if assigned_filter.key.name in ('assigned_to', 'assigned') else 'assigned_or_suggested'
        return f'{search_filter_key}:[{joined}]'
    return None

def update_saved_search_query(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    SavedSearch = apps.get_model('sentry', 'SavedSearch')
    assigned_regex = re.compile('(assigned|assigned_to|assigned_or_suggested):me($|\\s)', re.IGNORECASE)
    assigned_in_regex = re.compile('(assigned|assigned_to|assigned_or_suggested):\\[(.*?)]($|\\s)', re.IGNORECASE)
    for ss in RangeQuerySetWrapperWithProgressBar(SavedSearch.objects.all()):
        query = ss.query
        try:
            parse_search_query(query)
        except InvalidSearchQuery:
            continue
        assigned_me_idx_iter = re.finditer(assigned_regex, query)
        assigned_me_in_idx_iter = re.finditer(assigned_in_regex, query)
        all_idx = [m.span() for m in list(assigned_me_idx_iter) + list(assigned_me_in_idx_iter)]
        try:
            replacements = []
            for (start, stop) in all_idx or ():
                maybe_replacement = replacement_term(query[start:stop])
                if maybe_replacement:
                    replacements.append((start, stop, maybe_replacement))
            if replacements:
                result = []
                i = 0
                for (start, end, replacement) in replacements:
                    result.append(query[i:start] + replacement)
                    i = end
                result.append(query[i:])
                ss.query = ' '.join(result).strip()
                ss.save(update_fields=['query'])
        except Exception:
            continue

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0501_typed_bitfield_remove_labels')]
    operations = [migrations.RunPython(update_saved_search_query, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_savedsearch']})]