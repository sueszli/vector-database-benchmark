# Generated by Django 2.2.24 on 2022-05-03 15:43

import re

from django.db import migrations

from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

AGGREGATE_PATTERN = r"^(\w+)\((.*)?\)$"


def get_function_alias(field: str):
    match = re.match(AGGREGATE_PATTERN, field)

    if not match:
        return field

    if not match[2]:
        return match[1]

    arguments = parse_arguments(match[1], match[2])
    alias = match[1] + "_" + "_".join(arguments)

    for pattern in [r"[^\w]", r"^_+", r"_+$"]:
        alias = re.sub(pattern, "_", alias)

    return alias


def parse_arguments(function: str, columns: str):
    """
    Some functions take a quoted string for their arguments that may contain commas,
    which requires special handling.
    This function attempts to be identical with the similarly named parse_arguments
    found in static/app/utils/discover/fields.tsx
    """
    if (function != "to_other" and function != "count_if" and function != "spans_histogram") or len(
        columns
    ) == 0:
        return [c.strip() for c in columns.split(",") if len(c.strip()) > 0]

    args = []

    quoted = False
    escaped = False

    i, j = 0, 0

    while j < len(columns):
        if i == j and columns[j] == '"':
            # when we see a quote at the beginning of
            # an argument, then this is a quoted string
            quoted = True
        elif i == j and columns[j] == " ":
            # argument has leading spaces, skip over them
            i += 1
        elif quoted and not escaped and columns[j] == "\\":
            # when we see a slash inside a quoted string,
            # the next character is an escape character
            escaped = True
        elif quoted and not escaped and columns[j] == '"':
            # when we see a non-escaped quote while inside
            # of a quoted string, we should end it
            quoted = False
        elif quoted and escaped:
            # when we are inside a quoted string and have
            # begun an escape character, we should end it
            escaped = False
        elif quoted and columns[j] == ",":
            # when we are inside a quoted string and see
            # a comma, it should not be considered an
            # argument separator
            pass
        elif columns[j] == ",":
            # when we see a comma outside of a quoted string
            # it is an argument separator
            args.append(columns[i:j].strip())
            i = j + 1
        j += 1

    if i != j:
        # add in the last argument if any
        args.append(columns[i:].strip())

    return [arg for arg in args if arg]


def convert_dashboard_widget_query_orderby_to_function_format(apps, schema_editor):
    DashboardWidgetQuery = apps.get_model("sentry", "DashboardWidgetQuery")

    for query in RangeQuerySetWrapperWithProgressBar(DashboardWidgetQuery.objects.all()):
        if query.orderby == "":
            continue

        orderby = query.orderby
        stripped_orderby = orderby.lstrip("-")
        orderby_prefix = "-" if orderby.startswith("-") else ""

        for aggregate in query.aggregates:
            alias = get_function_alias(aggregate)
            if alias == stripped_orderby:
                query.orderby = f"{orderby_prefix}{aggregate}"
                query.save()
                continue


class Migration(CheckedMigration):
    # This flag is used to mark that a migration shouldn't be automatically run in production. For
    # the most part, this should only be used for operations where it's safe to run the migration
    # after your code has deployed. So this should not be used for most operations that alter the
    # schema of a table.
    # Here are some things that make sense to mark as dangerous:
    # - Large data migrations. Typically we want these to be run manually by ops so that they can
    #   be monitored and not block the deploy for a long period of time while they run.
    # - Adding indexes to large tables. Since this can take a long time, we'd generally prefer to
    #   have ops run this and not block the deploy. Note that while adding an index is a schema
    #   change, it's completely safe to run the operation after the code has deployed.
    is_dangerous = False

    # This flag is used to decide whether to run this migration in a transaction or not. Generally
    # we don't want to run in a transaction here, since for long running operations like data
    # back-fills this results in us locking an increasing number of rows until we finally commit.
    atomic = False

    dependencies = [
        ("sentry", "0288_fix_savedsearch_state"),
    ]

    operations = [
        migrations.RunPython(
            convert_dashboard_widget_query_orderby_to_function_format,
            migrations.RunPython.noop,
            hints={"tables": ["sentry_dashboardwidgetquery"]},
        ),
    ]
