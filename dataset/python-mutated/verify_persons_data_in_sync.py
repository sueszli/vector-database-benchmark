import json
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any, Dict, List
import structlog
from django.db.models.query import Prefetch
from django.utils.timezone import now
from posthog.celery import app
from posthog.client import sync_execute
from posthog.models.person import Person
logger = structlog.get_logger(__name__)
LIMIT = 100000
BATCH_SIZE = 500
PERIOD_START = timedelta(hours=1)
PERIOD_END = timedelta(days=2)
GET_PERSON_CH_QUERY = '\nSELECT id, version, properties FROM person JOIN (\n    SELECT id, max(version) as version, max(is_deleted) as is_deleted, team_id\n    FROM person\n    WHERE team_id IN %(team_ids)s AND id IN (%(person_ids)s)\n    GROUP BY team_id, id\n) as person_max ON person.id = person_max.id AND person.version = person_max.version AND person.team_id = person_max.team_id\nWHERE team_id IN %(team_ids)s\n  AND person_max.is_deleted = 0\n  AND id IN (%(person_ids)s)\n'
GET_DISTINCT_IDS_CH_QUERY = '\nSELECT distinct_id, argMax(person_id, version) as person_id\nFROM person_distinct_id2\nWHERE team_id IN %(team_ids)s\nGROUP BY team_id, distinct_id\nHAVING argMax(is_deleted, version) = 0 AND person_id IN (%(person_ids)s)\n'

@app.task(max_retries=1, ignore_result=True)
def verify_persons_data_in_sync(period_start: timedelta=PERIOD_START, period_end: timedelta=PERIOD_END, limit: int=LIMIT, emit_results: bool=True) -> Counter:
    if False:
        while True:
            i = 10
    max_pk = Person.objects.filter(created_at__lte=now() - period_start).latest('id').id
    person_data = list(Person.objects.filter(pk__lte=max_pk, pk__gte=max_pk - LIMIT * 5, created_at__gte=now() - period_end).values_list('id', 'uuid', 'team_id')[:limit])
    person_data.sort(key=lambda row: row[2])
    results = Counter({'total': 0, 'missing_in_clickhouse': 0, 'version_mismatch': 0, 'properties_mismatch': 0, 'distinct_ids_mismatch': 0, 'properties_mismatch_same_version': 0})
    for i in range(0, len(person_data), BATCH_SIZE):
        batch = person_data[i:i + BATCH_SIZE]
        results += _team_integrity_statistics(batch)
    if emit_results:
        _emit_metrics(results)
    return results

def _team_integrity_statistics(person_data: List[Any]) -> Counter:
    if False:
        return 10
    person_ids = [id for (id, _, _) in person_data]
    person_uuids = [uuid for (_, uuid, _) in person_data]
    team_ids = list(set((team_id for (_, _, team_id) in person_data)))
    pg_persons = _index_by(list(Person.objects.filter(id__in=person_ids).prefetch_related(Prefetch('persondistinctid_set', to_attr='distinct_ids_cache'))), lambda p: p.uuid)
    ch_persons = _index_by(sync_execute(GET_PERSON_CH_QUERY, {'person_ids': person_uuids, 'team_ids': team_ids}), lambda row: row[0])
    ch_distinct_ids_mapping = _index_by(sync_execute(GET_DISTINCT_IDS_CH_QUERY, {'person_ids': person_uuids, 'team_ids': team_ids}), lambda row: row[1], flat=False)
    result: Counter = Counter()
    for (_pk, uuid, team_id) in person_data:
        if uuid not in pg_persons:
            continue
        result['total'] += 1
        pg_person = pg_persons[uuid]
        if uuid not in ch_persons:
            result['missing_in_clickhouse'] += 1
            logger.info('Found person missing in clickhouse', team_id=team_id, uuid=uuid)
            continue
        (_, ch_version, ch_properties) = ch_persons[uuid]
        ch_properties = json.loads(ch_properties)
        if ch_version != pg_person.version:
            result['version_mismatch'] += 1
            logger.info('Found version mismatch', team_id=team_id, uuid=uuid, properties=pg_person.properties, ch_properties=ch_properties)
        if pg_person.properties != ch_properties:
            result['properties_mismatch'] += 1
            logger.info('Found properties mismatch', team_id=team_id, uuid=uuid, properties=pg_person.properties, ch_properties=ch_properties)
        if ch_version != 0 and ch_version == pg_person.version and (pg_person.properties != ch_properties):
            result['properties_mismatch_same_version'] += 1
        pg_distinct_ids = list(sorted(map(str, pg_person.distinct_ids)))
        ch_distinct_id = list(sorted((str(distinct_id) for (distinct_id, _) in ch_distinct_ids_mapping.get(uuid, []))))
        if pg_distinct_ids != ch_distinct_id:
            result['distinct_ids_mismatch'] += 1
    return result

def _emit_metrics(integrity_results: Counter) -> None:
    if False:
        return 10
    from statshog.defaults.django import statsd
    for (key, value) in integrity_results.items():
        statsd.gauge(f'posthog_person_integrity_{key}', value)

def _index_by(collection: List[Any], key_fn: Any, flat: bool=True) -> Dict:
    if False:
        i = 10
        return i + 15
    result: Dict = {} if flat else defaultdict(list)
    for item in collection:
        if flat:
            result[key_fn(item)] = item
        else:
            result[key_fn(item)].append(item)
    return result