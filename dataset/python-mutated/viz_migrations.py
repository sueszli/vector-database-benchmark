from enum import Enum
import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup
from flask.cli import with_appcontext
from superset import db

class VizType(str, Enum):
    AREA = 'area'
    DUAL_LINE = 'dual_line'
    LINE = 'line'
    PIVOT_TABLE = 'pivot_table'
    SUNBURST = 'sunburst'
    TREEMAP = 'treemap'

@click.group()
def migrate_viz() -> None:
    if False:
        while True:
            i = 10
    '\n    Migrate a viz from one type to another.\n    '

@migrate_viz.command()
@with_appcontext
@optgroup.group('Grouped options', cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option('--viz_type', '-t', help=f"The viz type to migrate: {', '.join(list(VizType))}")
def upgrade(viz_type: str) -> None:
    if False:
        while True:
            i = 10
    'Upgrade a viz to the latest version.'
    migrate(VizType(viz_type))

@migrate_viz.command()
@with_appcontext
@optgroup.group('Grouped options', cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option('--viz_type', '-t', help=f"The viz type to migrate: {', '.join(list(VizType))}")
def downgrade(viz_type: str) -> None:
    if False:
        while True:
            i = 10
    'Downgrade a viz to the previous version.'
    migrate(VizType(viz_type), is_downgrade=True)

def migrate(viz_type: VizType, is_downgrade: bool=False) -> None:
    if False:
        print('Hello World!')
    'Migrate a viz from one type to another.'
    from superset.migrations.shared.migrate_viz.processors import MigrateAreaChart, MigrateDualLine, MigrateLineChart, MigratePivotTable, MigrateSunburst, MigrateTreeMap
    migrations = {VizType.AREA: MigrateAreaChart, VizType.DUAL_LINE: MigrateDualLine, VizType.LINE: MigrateLineChart, VizType.PIVOT_TABLE: MigratePivotTable, VizType.SUNBURST: MigrateSunburst, VizType.TREEMAP: MigrateTreeMap}
    if is_downgrade:
        migrations[viz_type].downgrade(db.session)
    else:
        migrations[viz_type].upgrade(db.session)