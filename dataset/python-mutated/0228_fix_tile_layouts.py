import json
import structlog
from django.core.paginator import Paginator
from django.db import migrations

def migrate_dashboard_insight_relations(apps, _) -> None:
    if False:
        print('Hello World!')
    logger = structlog.get_logger(__name__)
    logger.info('starting_0228_fix_tile_layouts')
    DashboardTile = apps.get_model('posthog', 'DashboardTile')
    tiles = DashboardTile.objects.order_by('id').all()
    paginator = Paginator(tiles, 500)
    conversion_count = 0
    for page_number in paginator.page_range:
        page = paginator.page(page_number)
        updated_tiles = []
        for tile in page.object_list:
            if isinstance(tile.layouts, str):
                tile.layouts = json.loads(tile.layouts)
                if isinstance(tile.layouts, str):
                    tile.layouts = json.loads(tile.layouts)
                updated_tiles.append(tile)
        DashboardTile.objects.bulk_update(updated_tiles, ['layouts'])
        conversion_count += len(updated_tiles)
    logger.info('finished_0228_fix_tile_layouts', conversion_count=conversion_count)

class Migration(migrations.Migration):
    dependencies = [('posthog', '0227_add_dashboard_tiles')]

    def reverse(apps, _) -> None:
        if False:
            i = 10
            return i + 15
        pass
    operations = [migrations.RunPython(migrate_dashboard_insight_relations, reverse, elidable=True)]