from __future__ import annotations
from typing import Any
from django.db import connection, migrations
from psycopg2.extras import execute_values
from sentry_relay.exceptions import RelayError
from sentry_relay.processing import parse_release
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
BATCH_SIZE = 100

def convert_build_code_to_build_number(build_code):
    if False:
        while True:
            i = 10
    build_number = None
    if build_code is not None:
        try:
            build_code_as_int = int(build_code)
            if validate_bigint(build_code_as_int):
                build_number = build_code_as_int
        except ValueError:
            pass
    return build_number

def validate_bigint(value):
    if False:
        return 10
    return isinstance(value, int) and value >= 0 and (value.bit_length() <= 63)
UPDATE_QUERY = '\n    UPDATE sentry_release\n    SET package = data.package,\n    major = data.major::bigint,\n    minor = data.minor::bigint,\n    patch = data.patch::bigint,\n    revision = data.revision::bigint,\n    prerelease = data.prerelease,\n    build_code = data.build_code,\n    build_number = data.build_number::bigint\n    FROM (VALUES %s) AS data (id, package, major, minor, patch, revision, prerelease, build_code, build_number)\n    WHERE sentry_release.id = data.id'
SEMVER_FIELDS = ['package', 'major', 'minor', 'patch', 'revision', 'prerelease', 'build_code']

def backfill_semver(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Release = apps.get_model('sentry', 'Release')
    queryset = RangeQuerySetWrapperWithProgressBar(Release.objects.values_list('pk', 'version', 'package', 'major', 'minor', 'patch', 'revision', 'prerelease', 'build_code', 'build_number'), result_value_getter=lambda item: item[0])
    cursor = connection.cursor()
    batch: list[tuple[Any, ...]] = []
    for (pk, version, *semver_fields) in queryset:
        try:
            version_info = parse_release(version)
        except RelayError:
            continue
        version_parsed = version_info.get('version_parsed')
        if version_parsed is None:
            if semver_fields[0] is None:
                continue
            batch.append((pk, None, None, None, None, None, None, None, None))
        else:
            bigint_fields = ['major', 'minor', 'patch', 'revision']
            if not all((validate_bigint(version_parsed[field]) for field in bigint_fields)):
                continue
            build_code = version_parsed.get('build_code')
            build_number = convert_build_code_to_build_number(build_code)
            new_vals = [version_info['package'], version_parsed['major'], version_parsed['minor'], version_parsed['patch'], version_parsed['revision'], version_parsed['pre'] or '', build_code, build_number]
            if semver_fields != new_vals:
                batch.append((pk, *new_vals))
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0231_alert_rule_comparison_delta')]
    operations = [migrations.RunPython(backfill_semver, migrations.RunPython.noop, hints={'tables': ['sentry_release']})]