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
UPDATE_QUERY = '\n    UPDATE sentry_release\n    SET package = data.package,\n    major = data.major,\n    minor = data.minor,\n    patch = data.patch,\n    revision = data.revision,\n    prerelease = data.prerelease,\n    build_code = data.build_code,\n    build_number = data.build_number::bigint\n    FROM (VALUES %s) AS data (id, package, major, minor, patch, revision, prerelease, build_code, build_number)\n    WHERE sentry_release.id = data.id'

def backfill_semver(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Release = apps.get_model('sentry', 'Release')
    queryset = RangeQuerySetWrapperWithProgressBar(Release.objects.values_list('pk', 'version'), result_value_getter=lambda item: item[0])
    cursor = connection.cursor()
    batch = []
    for (pk, version) in queryset:
        try:
            version_info = parse_release(version)
        except RelayError:
            continue
        version_parsed = version_info.get('version_parsed')
        if version_parsed is None:
            continue
        bigint_fields = ['major', 'minor', 'patch', 'revision']
        if not all((validate_bigint(version_parsed[field]) for field in bigint_fields)):
            continue
        build_code = version_parsed.get('build_code')
        build_number = convert_build_code_to_build_number(build_code)
        batch.append((pk, version_info['package'], version_parsed['major'], version_parsed['minor'], version_parsed['patch'], version_parsed['revision'], version_parsed['pre'] or '', build_code, build_number))
        if len(batch) >= BATCH_SIZE:
            execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)
            batch = []
    if batch:
        execute_values(cursor, UPDATE_QUERY, batch, page_size=BATCH_SIZE)

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0204_use_project_team_for_team_key_transactions')]
    operations = [migrations.RunPython(backfill_semver, migrations.RunPython.noop, hints={'tables': ['sentry_release']})]