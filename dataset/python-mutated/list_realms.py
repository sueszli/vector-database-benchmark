import sys
from argparse import ArgumentParser
from typing import Any
from typing_extensions import override
from zerver.lib.management import ZulipBaseCommand
from zerver.models import Realm

class Command(ZulipBaseCommand):
    help = "List realms in the server and it's configuration settings(optional).\n\nUsage examples:\n\n./manage.py list_realms\n./manage.py list_realms --all"

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            for i in range(10):
                print('nop')
        parser.add_argument('--all', action='store_true', help='Print all the configuration settings of the realms.')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        realms = Realm.objects.all()
        outer_format = '{:<5} {:<20} {!s:<30} {:<50}'
        inner_format = '{:<40} {}'
        deactivated = False
        if not options['all']:
            print(outer_format.format('id', 'string_id', 'name', 'domain'))
            print(outer_format.format('--', '---------', '----', '------'))
            for realm in realms:
                display_string_id = realm.string_id if realm.string_id != '' else "''"
                if realm.deactivated:
                    print(self.style.ERROR(outer_format.format(realm.id, display_string_id, realm.name, realm.uri)))
                    deactivated = True
                else:
                    print(outer_format.format(realm.id, display_string_id, realm.name, realm.uri))
            if deactivated:
                print(self.style.WARNING('\nRed rows represent deactivated realms.'))
            sys.exit(0)
        identifier_attributes = ['id', 'name', 'string_id']
        for realm in realms:
            realm_dict = vars(realm).copy()
            del realm_dict['_state']
            realm_dict['authentication_methods'] = str(realm.authentication_methods_dict())
            for key in identifier_attributes:
                if realm.deactivated:
                    print(self.style.ERROR(inner_format.format(key, realm_dict[key])))
                    deactivated = True
                else:
                    print(inner_format.format(key, realm_dict[key]))
            for (key, value) in sorted(realm_dict.items()):
                if key not in identifier_attributes:
                    if realm.deactivated:
                        print(self.style.ERROR(inner_format.format(key, value)))
                    else:
                        print(inner_format.format(key, value))
            print('-' * 80)
        if deactivated:
            print(self.style.WARNING('\nRed is used to highlight deactivated realms.'))