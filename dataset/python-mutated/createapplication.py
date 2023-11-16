from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand
from oauth2_provider.models import get_application_model
Application = get_application_model()

class Command(BaseCommand):
    help = 'Shortcut to create a new application in a programmatic way'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('client_type', type=str, help='The client type, one of: %s' % ', '.join([ctype[0] for ctype in Application.CLIENT_TYPES]))
        parser.add_argument('authorization_grant_type', type=str, help='The type of authorization grant to be used, one of: %s' % ', '.join([gtype[0] for gtype in Application.GRANT_TYPES]))
        parser.add_argument('--client-id', type=str, help='The ID of the new application')
        parser.add_argument('--user', type=str, help='The user the application belongs to')
        parser.add_argument('--redirect-uris', type=str, help="The redirect URIs, this must be a space separated string e.g 'URI1 URI2'")
        parser.add_argument('--post-logout-redirect-uris', type=str, help="The post logout redirect URIs, this must be a space separated string e.g 'URI1 URI2'", default='')
        parser.add_argument('--client-secret', type=str, help='The secret for this application')
        parser.add_argument('--no-hash-client-secret', dest='hash_client_secret', action='store_false', help="Don't hash the client secret")
        parser.set_defaults(hash_client_secret=True)
        parser.add_argument('--name', type=str, help='The name this application')
        parser.add_argument('--skip-authorization', action='store_true', help='If set, completely bypass the authorization form, even on the first use of the application')
        parser.add_argument('--algorithm', type=str, help='The OIDC token signing algorithm for this application, one of: %s' % ', '.join([atype[0] for atype in Application.ALGORITHM_TYPES if atype[0]]))

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        application_fields = [field.name for field in Application._meta.fields]
        application_data = {}
        for (key, value) in options.items():
            if key in application_fields and (isinstance(value, bool) or value):
                if key == 'user':
                    application_data.update({'user_id': value})
                else:
                    application_data.update({key: value})
        new_application = Application(**application_data)
        try:
            new_application.full_clean()
        except ValidationError as exc:
            errors = '\n '.join(['- ' + err_key + ': ' + str(err_value) for (err_key, err_value) in exc.message_dict.items()])
            self.stdout.write(self.style.ERROR('Please correct the following errors:\n %s' % errors))
        else:
            cleartext_secret = new_application.client_secret
            new_application.save()
            client_name_or_id = application_data.get('name', new_application.client_id)
            self.stdout.write(self.style.SUCCESS('New application %s created successfully.' % client_name_or_id))
            if 'client_secret' not in application_data:
                self.stdout.write(self.style.SUCCESS('client_secret: %s' % cleartext_secret))