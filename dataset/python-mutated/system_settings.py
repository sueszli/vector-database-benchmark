from django.core.management.base import BaseCommand
from dojo.models import System_Settings

class Command(BaseCommand):
    help = 'Updates product grade calculation'

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        code = 'def grade_product(crit, high, med, low):\n            health=100\n            if crit > 0:\n                health = 40\n                health = health - ((crit - 1) * 5)\n            if high > 0:\n                if health == 100:\n                    health = 60\n                health = health - ((high - 1) * 3)\n            if med > 0:\n                if health == 100:\n                    health = 80\n                health = health - ((med - 1) * 2)\n            if low > 0:\n                if health == 100:\n                    health = 95\n                health = health - low\n\n            if health < 5:\n                health = 5\n\n            return health\n            '
        system_settings = System_Settings.objects.get(id=1)
        system_settings.product_grade = code
        system_settings.save()