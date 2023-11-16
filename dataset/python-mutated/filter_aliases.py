from __future__ import annotations
from functools import reduce
from typing import List, Mapping, Optional
from snuba_sdk import Condition, Op
from sentry.api.event_search import SearchFilter, SearchKey, SearchValue
from sentry.exceptions import InvalidSearchQuery
from sentry.models.release import Release, SemverFilter
from sentry.search.events import builder, constants
from sentry.search.events.filter import _flip_field_sort, handle_operator_negation, parse_semver, to_list
from sentry.search.events.types import WhereType
from sentry.search.utils import DEVICE_CLASS, parse_release
from sentry.utils.strings import oxfordize_list

def team_key_transaction_filter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> WhereType:
    if False:
        while True:
            i = 10
    value = search_filter.value.value
    key_transaction_expr = builder.resolve_field_alias(constants.TEAM_KEY_TRANSACTION_ALIAS)
    if search_filter.value.raw_value == '':
        return Condition(key_transaction_expr, Op.NEQ if search_filter.operator == '!=' else Op.EQ, 0)
    if value in ('1', 1):
        return Condition(key_transaction_expr, Op.EQ, 1)
    if value in ('0', 0):
        return Condition(key_transaction_expr, Op.EQ, 0)
    raise InvalidSearchQuery('Invalid value for key_transaction condition. Accepted values are 1, 0')

def release_filter_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        i = 10
        return i + 15
    'Parse releases for potential aliases like `latest`'
    if search_filter.value.is_wildcard():
        operator = search_filter.operator
        value = search_filter.value
    else:
        operator_conversions = {'=': 'IN', '!=': 'NOT IN'}
        operator = operator_conversions.get(search_filter.operator, search_filter.operator)
        value = SearchValue(reduce(lambda x, y: x + y, [parse_release(v, builder.params.project_ids, builder.params.environments, builder.params.organization.id if builder.params.organization else None) for v in to_list(search_filter.value.value)], []))
    return builder.default_filter_converter(SearchFilter(search_filter.key, operator, value))

def project_slug_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        while True:
            i = 10
    'Convert project slugs to ids and create a filter based on those.\n    This is cause we only store project ids in clickhouse.\n    '
    value = search_filter.value.value
    if Op(search_filter.operator) == Op.EQ and value == '':
        raise InvalidSearchQuery('Cannot query for has:project or project:"" as every event will have a project')
    slugs = to_list(value)
    project_slugs: Mapping[str, int] = {slug: project_id for (slug, project_id) in builder.params.project_slug_map.items() if slug in slugs}
    missing: List[str] = [slug for slug in slugs if slug not in project_slugs]
    if missing and search_filter.operator in constants.EQUALITY_OPERATORS:
        raise InvalidSearchQuery(f'Invalid query. Project(s) {oxfordize_list(missing)} do not exist or are not actively selected.')
    project_ids = list(sorted(project_slugs.values()))
    if project_ids:
        converted_filter = builder.convert_search_filter_to_condition(SearchFilter(SearchKey('project.id'), search_filter.operator, SearchValue(project_ids if search_filter.is_in_filter else project_ids[0])))
        if converted_filter:
            if search_filter.operator in constants.EQUALITY_OPERATORS:
                builder.projects_to_filter.update(project_ids)
            return converted_filter
    return None

def release_stage_filter_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        return 10
    '\n    Parses a release stage search and returns a snuba condition to filter to the\n    requested releases.\n    '
    if builder.params.organization is None:
        raise ValueError('organization is a required param')
    qs = Release.objects.filter_by_stage(builder.params.organization.id, search_filter.operator, search_filter.value.value, project_ids=builder.params.project_ids, environments=builder.params.environments).values_list('version', flat=True).order_by('date_added')[:constants.MAX_SEARCH_RELEASES]
    versions = list(qs)
    if not versions:
        versions = [constants.SEMVER_EMPTY_RELEASE]
    return Condition(builder.column('release'), Op.IN, versions)

def semver_filter_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Parses a semver query search and returns a snuba condition to filter to the\n    requested releases.\n\n    Since we only have semver information available in Postgres currently, we query\n    Postgres and return a list of versions to include/exclude. For most customers this\n    will work well, however some have extremely large numbers of releases, and we can't\n    pass them all to Snuba. To try and serve reasonable results, we:\n     - Attempt to query based on the initial semver query. If this returns\n       MAX_SEMVER_SEARCH_RELEASES results, we invert the query and see if it returns\n       fewer results. If so, we use a `NOT IN` snuba condition instead of an `IN`.\n     - Order the results such that the versions we return are semantically closest to\n       the passed filter. This means that when searching for `>= 1.0.0`, we'll return\n       version 1.0.0, 1.0.1, 1.1.0 before 9.x.x.\n    "
    if builder.params.organization is None:
        raise ValueError('organization is a required param')
    organization_id: int = builder.params.organization.id
    version: str = search_filter.value.raw_value
    operator: str = search_filter.operator
    order_by = Release.SEMVER_COLS
    if operator.startswith('<'):
        order_by = list(map(_flip_field_sort, order_by))
    qs = Release.objects.filter_by_semver(organization_id, parse_semver(version, operator), project_ids=builder.params.project_ids).values_list('version', flat=True).order_by(*order_by)[:constants.MAX_SEARCH_RELEASES]
    versions = list(qs)
    final_operator = Op.IN
    if len(versions) == constants.MAX_SEARCH_RELEASES:
        operator = constants.OPERATOR_NEGATION_MAP[operator]
        qs_flipped = Release.objects.filter_by_semver(organization_id, parse_semver(version, operator)).order_by(*map(_flip_field_sort, order_by)).values_list('version', flat=True)[:constants.MAX_SEARCH_RELEASES]
        exclude_versions = list(qs_flipped)
        if exclude_versions and len(exclude_versions) < len(versions):
            final_operator = Op.NOT_IN
            versions = exclude_versions
    if not versions:
        versions = [constants.SEMVER_EMPTY_RELEASE]
    return Condition(builder.column('release'), final_operator, versions)

def semver_package_filter_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        while True:
            i = 10
    '\n    Applies a semver package filter to the search. Note that if the query returns more than\n    `MAX_SEARCH_RELEASES` here we arbitrarily return a subset of the releases.\n    '
    if builder.params.organization is None:
        raise ValueError('organization is a required param')
    package: str = search_filter.value.raw_value
    versions = list(Release.objects.filter_by_semver(builder.params.organization.id, SemverFilter('exact', [], package), project_ids=builder.params.project_ids).values_list('version', flat=True)[:constants.MAX_SEARCH_RELEASES])
    if not versions:
        versions = [constants.SEMVER_EMPTY_RELEASE]
    return Condition(builder.column('release'), Op.IN, versions)

def semver_build_filter_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        i = 10
        return i + 15
    '\n    Applies a semver build filter to the search. Note that if the query returns more than\n    `MAX_SEARCH_RELEASES` here we arbitrarily return a subset of the releases.\n    '
    if builder.params.organization is None:
        raise ValueError('organization is a required param')
    build: str = search_filter.value.raw_value
    (operator, negated) = handle_operator_negation(search_filter.operator)
    try:
        django_op = constants.OPERATOR_TO_DJANGO[operator]
    except KeyError:
        raise InvalidSearchQuery("Invalid operation 'IN' for semantic version filter.")
    versions = list(Release.objects.filter_by_semver_build(builder.params.organization.id, django_op, build, project_ids=builder.params.project_ids, negated=negated).values_list('version', flat=True)[:constants.MAX_SEARCH_RELEASES])
    if not versions:
        versions = [constants.SEMVER_EMPTY_RELEASE]
    return Condition(builder.column('release'), Op.IN, versions)

def device_class_converter(builder: builder.QueryBuilder, search_filter: SearchFilter) -> Optional[WhereType]:
    if False:
        return 10
    value = search_filter.value.value
    if value not in DEVICE_CLASS:
        raise InvalidSearchQuery(f'{value} is not a supported device.class')
    return Condition(builder.column('device.class'), Op.IN, list(DEVICE_CLASS[value]))