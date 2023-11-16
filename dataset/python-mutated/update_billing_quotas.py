import pprint
from django.core.management.base import BaseCommand
from ee.billing.quota_limiting import update_all_org_billing_quotas

class Command(BaseCommand):
    help = 'Update billing rate limiting for all organizations'

    def add_arguments(self, parser):
        if False:
            for i in range(10):
                print('nop')
        parser.add_argument('--dry-run', type=bool, help='Print information instead of storing it')
        parser.add_argument('--print-reports', type=bool, help='Print the reports in full')

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        dry_run = options['dry_run']
        results = update_all_org_billing_quotas(dry_run)
        if options['print_reports']:
            print('')
            pprint.pprint(results)
            print('')
        if dry_run:
            print('Dry run so not stored.')
        else:
            print(f"{len(results['events'])} orgs rate limited for events")
            print(f"{len(results['recordings'])} orgs rate limited for recordings")
            print('Done!')